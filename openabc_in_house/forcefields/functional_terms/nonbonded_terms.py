import numpy as np
import pandas as pd
import simtk.openmm as mm
import simtk.openmm.app as app
import simtk.unit as unit
import math
import sys
import os

'''
Note addGlobalParameter can automatically convert the values to the correct unit. 
Be careful that addGlobalParameter sets global parameters that is used by all the forces in the system. 
'''

# define some constants based on CODATA
NA = unit.AVOGADRO_CONSTANT_NA # Avogadro constant
kB = unit.BOLTZMANN_CONSTANT_kB  # Boltzmann constant
EC = 1.602176634e-19*unit.coulomb # elementary charge
VEP = 8.8541878128e-12*unit.farad/unit.meter # vacuum electric permittivity

_amino_acids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                'SER', 'THR', 'TRP', 'TYR', 'VAL']

_nucleotides = ['DA', 'DT', 'DC', 'DG']



def moff_mrg_contact_term(atom_types, df_exclusions, use_pbc, alpha_map, epsilon_map, eta=0.7/unit.angstrom, 
                          r0=8.0*unit.angstrom, cutoff=2.0*unit.nanometer, force_group=5):
    '''
    MOFF+MRG model nonbonded contact term.
    '''
    eta_value = eta.value_in_unit(unit.nanometer**-1)
    r0_value = r0.value_in_unit(unit.nanometer)
    cutoff_value = cutoff.value_in_unit(unit.nanometer)
    contacts = mm.CustomNonbondedForce(f'''energy;
               energy=(energy1+energy2-offset1-offset2)*step({cutoff_value}-r);
               energy1=alpha_con/(r^12);
               energy2=-0.5*epsilon_con*(1+tanh({eta_value}*({r0_value}-r)));
               offset1=alpha_con/({cutoff_value}^12);
               offset2=-0.5*epsilon_con*(1+tanh({eta_value}*({r0_value}-{cutoff_value})));
               alpha_con=alpha_con_map(atom_type1, atom_type2);
               epsilon_con=epsilon_con_map(atom_type1, atom_type2);
               ''')
    n_atom_types = alpha_map.shape[0]
    discrete_2d_alpha_map = mm.Discrete2DFunction(n_atom_types, n_atom_types, alpha_map.ravel().tolist())
    discrete_2d_epsilon_map = mm.Discrete2DFunction(n_atom_types, n_atom_types, epsilon_map.ravel().tolist())
    contacts.addTabulatedFunction('alpha_con_map', discrete_2d_alpha_map)
    contacts.addTabulatedFunction('epsilon_con_map', discrete_2d_epsilon_map)
    contacts.addPerParticleParameter('atom_type')
    for each in atom_types:
        contacts.addParticle([each])
    for i, row in df_exclusions.iterrows():
        contacts.addExclusion(int(row['a1']), int(row['a2']))
    if use_pbc:
        contacts.setNonbondedMethod(contacts.CutoffPeriodic)
    else:
        contacts.setNonbondedMethod(contacts.CutoffNonPeriodic)
    contacts.setCutoffDistance(cutoff)
    contacts.setForceGroup(force_group)
    return contacts


def hps_ah_term(atom_types, df_exclusions, use_pbc, epsilon, sigma_ah_map, lambda_ah_map, force_group=2):
    '''
    HPS model nonbonded contact term (form proposed by Ashbaugh and Hatch). 
    The cutoff is 4*sigma_ah. 
    '''
    lj_at_cutoff = 4*epsilon*((1/4)**12 - (1/4)**6)
    contacts = mm.CustomNonbondedForce(f'''energy;
               energy=(f1+f2-offset)*step(4*sigma_ah-r);
               offset=lambda_ah*{lj_at_cutoff};
               f1=(lj+(1-lambda_ah)*{epsilon})*step(2^(1/6)*sigma_ah-r);
               f2=lambda_ah*lj*step(r-2^(1/6)*sigma_ah);
               lj=4*{epsilon}*((sigma_ah/r)^12-(sigma_ah/r)^6);
               sigma_ah=sigma_ah_map(atom_type1, atom_type2);
               lambda_ah=lambda_ah_map(atom_type1, atom_type2);
               ''')
    n_atom_types = sigma_ah_map.shape[0]
    discrete_2d_sigma_ah_map = mm.Discrete2DFunction(n_atom_types, n_atom_types, sigma_ah_map.ravel().tolist())
    discrete_2d_lambda_ah_map = mm.Discrete2DFunction(n_atom_types, n_atom_types, lambda_ah_map.ravel().tolist())
    contacts.addTabulatedFunction('sigma_ah_map', discrete_2d_sigma_ah_map)
    contacts.addTabulatedFunction('lambda_ah_map', discrete_2d_lambda_ah_map)
    contacts.addPerParticleParameter('atom_type')
    for each in atom_types:
        contacts.addParticle([each])
    for i, row in df_exclusions.iterrows():
        contacts.addExclusion(int(row['a1']), int(row['a2']))
    if use_pbc:
        contacts.setNonbondedMethod(contacts.CutoffPeriodic)
    else:
        contacts.setNonbondedMethod(contacts.CutoffNonPeriodic)
    contacts.setCutoffDistance(4*np.amax(sigma_ah_map))
    contacts.setForceGroup(force_group)
    return contacts


def ddd_dh_elec_term(charges, df_exclusions, use_pbc, salt_conc=150.0*unit.millimolar, 
                     temperature=300.0*unit.kelvin, cutoff=4.0*unit.nanometer, force_group=6):
    '''
    Debye-Huckel potential with a distance-dependent dielectric.
    '''
    alpha = NA*EC**2/(4*np.pi*VEP)
    gamma = VEP*kB*temperature/(2.0*NA*salt_conc*EC**2)
    # use a distance-dependent relative permittivity (dielectric)
    dielectric_water = 78.4
    A = -8.5525
    kappa = 7.7839
    B = dielectric_water - A
    zeta = 0.03627
    cutoff_value = cutoff.value_in_unit(unit.nanometer)
    alpha_value = alpha.value_in_unit(unit.kilojoule_per_mole*unit.nanometer)
    gamma_value = gamma.value_in_unit(unit.nanometer**2)
    dielectric_at_cutoff = A + B/(1 + kappa*math.exp(-zeta*B*cutoff_value))
    ldby_at_cutoff = (dielectric_at_cutoff*gamma_value)**0.5
    elec = mm.CustomNonbondedForce(f'''energy;
           energy=q1*q2*{alpha_value}*((exp(-r/ldby)/r)-offset)*step({cutoff_value}-r)/dielectric;
           offset={math.exp(-cutoff_value/ldby_at_cutoff)/cutoff_value};
           ldby=(dielectric*{gamma_value})^0.5;
           dielectric={A}+{B}/(1+{kappa}*exp(-{zeta}*{B}*r));
           ''')
    elec.addPerParticleParameter('q')
    for q in charges:
        elec.addParticle([q])
    for i, row in df_exclusions.iterrows():
        elec.addExclusion(int(row['a1']), int(row['a2']))
    if use_pbc:
        elec.setNonbondedMethod(elec.CutoffPeriodic)
    else:
        elec.setNonbondedMethod(elec.CutoffNonPeriodic)
    elec.setCutoffDistance(cutoff)
    elec.setForceGroup(force_group)
    return elec
    

def ddd_dh_elec_switch_term(charges, df_exclusions, use_pbc, salt_conc=150.0*unit.millimolar, 
                            temperature=300.0*unit.kelvin, cutoff1=1.2*unit.nanometer, cutoff2=1.5*unit.nanometer, 
                            switch_coeff=[1, 0, 0, -10, 15, -6], force_group=6):
    '''
    Debye-Huckel potential with a distance-dependent dielectric and a switch function. 
    The switch function value changes from 1 to 0 smoothly as distance r changes from cutoff1 to cutoff2. 
    To make sure the switch function works properly, the zeroth order coefficient has to be 1, and the sum of all the coefficients in switch_coeff has to be 0. 
    '''
    alpha = NA*EC**2/(4*np.pi*VEP)
    gamma = VEP*kB*temperature/(2.0*NA*salt_conc*EC**2)
    # use a distance-dependent relative permittivity (dielectric)
    dielectric_water = 78.4
    A = -8.5525
    kappa = 7.7839
    B = dielectric_water - A
    zeta = 0.03627
    alpha_value = alpha.value_in_unit(unit.kilojoule_per_mole*unit.nanometer)
    cutoff1_value = cutoff1.value_in_unit(unit.nanometer)
    cutoff2_value = cutoff2.value_in_unit(unit.nanometer)
    gamma_value = gamma.value_in_unit(unit.nanometer**2)
    assert switch_coeff[0] == 1
    assert np.sum(np.array(switch_coeff)) == 0
    switch_term_list = []
    for i in range(len(switch_coeff)):
        if i == 0:
            switch_term_list.append(f'{switch_coeff[i]}')
        else:
            switch_term_list.append(f'({switch_coeff[i]}*((r-{cutoff1_value})/({cutoff2_value}-{cutoff1_value}))^{i})')
    switch_term_string = '+'.join(switch_term_list)
    elec = mm.CustomNonbondedForce(f'''energy;
           energy=q1*q2*{alpha_value}*exp(-r/ldby)*switch/(dielectric*r);
           switch=({switch_term_string})*step(r-{cutoff1_value})*step({cutoff2_value}-r)+step({cutoff1_value}-r);
           ldby=(dielectric*{gamma_value})^0.5;
           dielectric={A}+{B}/(1+{kappa}*exp(-{zeta}*{B}*r));
           ''')
    elec.addPerParticleParameter('q')
    for q in charges:
        elec.addParticle([q])
    for i, row in df_exclusions.iterrows():
        elec.addExclusion(int(row['a1']), int(row['a2']))
    if use_pbc:
        elec.setNonbondedMethod(elec.CutoffPeriodic)
    else:
        elec.setNonbondedMethod(elec.CutoffNonPeriodic)
    elec.setCutoffDistance(cutoff2)
    elec.setForceGroup(force_group)
    return elec


def dh_elec_term(charges, df_exclusions, use_pbc, ldby=1*unit.nanometer, dielectric_water=80.0, 
                 cutoff=3.5*unit.nanometer, force_group=3):
    '''
    Debye-Huckel potential with a constant dielectric.
    '''
    alpha = NA*EC**2/(4*np.pi*VEP)
    ldby_value = ldby.value_in_unit(unit.nanometer)
    alpha_value = alpha.value_in_unit(unit.kilojoule_per_mole*unit.nanometer)
    cutoff_value = cutoff.value_in_unit(unit.nanometer)
    elec = mm.CustomNonbondedForce(f'''energy;
           energy=q1*q2*{alpha_value}*((exp(-r/{ldby_value})/r)-offset)*step({cutoff_value}-r)/{dielectric_water};
           offset={math.exp(-cutoff_value/ldby_value)/cutoff_value};
           ''')
    elec.addPerParticleParameter('q')
    for q in charges:
        elec.addParticle([q])
    for i, row in df_exclusions.iterrows():
        elec.addExclusion(int(row['a1']), int(row['a2']))
    if use_pbc:
        elec.setNonbondedMethod(elec.CutoffPeriodic)
    else:
        elec.setNonbondedMethod(elec.CutoffNonPeriodic)
    elec.setCutoffDistance(cutoff)
    elec.setForceGroup(force_group)
    return elec


def ddd_dh_elec_switch_term_map(charges, df_exclusions, use_pbc, salt_conc=150.0*unit.millimolar, 
                            temperature=300.0*unit.kelvin, cutoff1=1.2*unit.nanometer, cutoff2=1.5*unit.nanometer, 
                            switch_coeff=[1, 0, 0, -10, 15, -6], force_group=6):
    '''
    Debye-Huckel potential with a distance-dependent dielectric and a switch function. 
    The switch function value changes from 1 to 0 smoothly as distance r changes from cutoff1 to cutoff2. 
    To make sure the switch function works properly, the zeroth order coefficient has to be 1, and the sum of all the coefficients in switch_coeff has to be 0. 
    '''
    alpha = NA*EC**2/(4*np.pi*VEP)
    gamma = VEP*kB*temperature/(2.0*NA*salt_conc*EC**2)
    # use a distance-dependent relative permittivity (dielectric)
    dielectric_water = 78.4
    A = -8.5525
    kappa = 7.7839
    B = dielectric_water - A
    zeta = 0.03627
    alpha_value = alpha.value_in_unit(unit.kilojoule_per_mole*unit.nanometer)
    cutoff1_value = cutoff1.value_in_unit(unit.nanometer)
    cutoff2_value = cutoff2.value_in_unit(unit.nanometer)
    gamma_value = gamma.value_in_unit(unit.nanometer**2)
    assert switch_coeff[0] == 1
    assert np.sum(np.array(switch_coeff)) == 0
    switch_term_list = []
    for i in range(len(switch_coeff)):
        if i == 0:
            switch_term_list.append(f'{switch_coeff[i]}')
        else:
            switch_term_list.append(f'({switch_coeff[i]}*((r-{cutoff1_value})/({cutoff2_value}-{cutoff1_value}))^{i})')
    switch_term_string = '+'.join(switch_term_list)
    elec = mm.CustomNonbondedForce(f'''energy;
           energy=q1*q2*{alpha_value}*exp(-r/ldby)*switch/(dielectric*r);
           switch=({switch_term_string})*step(r-{cutoff1_value})*step({cutoff2_value}-r)+step({cutoff1_value}-r);
           ldby=(dielectric*{gamma_value})^0.5;
           dielectric={A}+{B}/(1+{kappa}*exp(-{zeta}*{B}*r));
           ''')
    elec.addPerParticleParameter('q')
    for q in charges:
        elec.addParticle([q])
    for i, row in df_exclusions.iterrows():
        elec.addExclusion(int(row['a1']), int(row['a2']))
    if use_pbc:
        elec.setNonbondedMethod(elec.CutoffPeriodic)
    else:
        elec.setNonbondedMethod(elec.CutoffNonPeriodic)
    elec.setCutoffDistance(cutoff2)
    elec.setForceGroup(force_group)
    return elec


def ddd_dh_elec_switch_term_map(mol, salt_conc=150.0*unit.millimolar, 
                            temperature=300.0*unit.kelvin, cutoff1=1.2*unit.nanometer, cutoff2=1.5*unit.nanometer, 
                            switch_coeff=[1, 0, 0, -10, 15, -6], force_group=6):
    '''
    Debye-Huckel potential with a distance-dependent dielectric and a switch function. 
    The switch function value changes from 1 to 0 smoothly as distance r changes from cutoff1 to cutoff2. 
    To make sure the switch function works properly, the zeroth order coefficient has to be 1, and the sum of all the coefficients in switch_coeff has to be 0. 
    

    CG atom types: 
    Type 0 for protein CG atoms. 
    Type 1 for DNA CG atoms.      
    
    '''
    alpha = NA*EC**2/(4*np.pi*VEP)
    gamma = VEP*kB*temperature/(2.0*NA*salt_conc*EC**2)
    # use a distance-dependent relative permittivity (dielectric)
    dielectric_water = 78.4
    A = -8.5525
    kappa = 7.7839
    B = dielectric_water - A
    zeta = 0.03627
    alpha_value = alpha.value_in_unit(unit.kilojoule_per_mole*unit.nanometer)
    cutoff1_value = cutoff1.value_in_unit(unit.nanometer)
    cutoff2_value = cutoff2.value_in_unit(unit.nanometer)
    gamma_value = gamma.value_in_unit(unit.nanometer**2)

    # For mapping switching functions differently for p-p, p-n, and n-n interactions;
    # p-p switch, p-n no switch, n-n no switch
    # amino acids: atomtype = 0, nucleotides: atomtype = 1;
    n_atom_types = 2

    # Switch function coefficient for each atom type
    switch_coeff1_map = np.zeros((n_atom_types, n_atom_types))
    switch_coeff2_map = np.zeros((n_atom_types, n_atom_types))
    switch_coeff3_map = np.zeros((n_atom_types, n_atom_types))
    switch_coeff4_map = np.zeros((n_atom_types, n_atom_types))
    switch_coeff5_map = np.zeros((n_atom_types, n_atom_types))

    # Manning function for each atom type;
    # p-p = p-n = 1.0, n-n = 0.36;
    manning_coeff_map = np.zeros((n_atom_types, n_atom_types))

    # Cutoff for each atom type
    cutoff_r1_map = np.zeros((n_atom_types, n_atom_types))
    cutoff_r2_map = np.zeros((n_atom_types, n_atom_types))


    for idx in range(0, 2, 1):
        for jdx in range(0, 2, 1):

            if (idx == 0 and jdx == 0):
                switch_coeff1_map[idx, jdx] = switch_coeff[1]
                switch_coeff2_map[idx, jdx] = switch_coeff[2]
                switch_coeff3_map[idx, jdx] = switch_coeff[3]
                switch_coeff4_map[idx, jdx] = switch_coeff[4]
                switch_coeff5_map[idx, jdx] = switch_coeff[5]
                # Set the manning coefficient = 1.0
                manning_coeff_map[idx, jdx] = 1.0
                cutoff_r1_map[idx, jdx] = cutoff1_value
                cutoff_r2_map[idx, jdx] = cutoff2_value
            elif(idx == 1 and jdx == 1):
                switch_coeff1_map[idx, jdx] = 0.0
                switch_coeff2_map[idx, jdx] = 0.0
                switch_coeff3_map[idx, jdx] = 0.0
                switch_coeff4_map[idx, jdx] = 0.0
                switch_coeff5_map[idx, jdx] = 0.0
                # Set the manning coefficient = 0.36, mimicking Manning codensation in the dna-dna interaction
                manning_coeff_map[idx, jdx] = 0.36
                # here, we will set S(r) to be 1.0 from 0 to r2, so r1 can be set as an <r2 arbitrary value
                cutoff_r1_map[idx, jdx] = 1.5
                # r2 set to 5 DH length;
                cutoff_r2_map[idx, jdx] = 4.0                
            else:
                switch_coeff1_map[idx, jdx] = 0.0
                switch_coeff2_map[idx, jdx] = 0.0
                switch_coeff3_map[idx, jdx] = 0.0
                switch_coeff4_map[idx, jdx] = 0.0
                switch_coeff5_map[idx, jdx] = 0.0
                # Set the manning coefficient = 1.0
                manning_coeff_map[idx, jdx] = 1.0
                # here, we will set S(r) to be 1.0 from 0 to r2, so r1 can be set as an <r2 arbitrary value
                cutoff_r1_map[idx, jdx] = 1.5
                # r2 set to 5 DH length;
                cutoff_r2_map[idx, jdx] = 4.0

    switch_coeff1_map = switch_coeff1_map.ravel().tolist() 
    switch_coeff2_map = switch_coeff2_map.ravel().tolist() 
    switch_coeff3_map = switch_coeff3_map.ravel().tolist() 
    switch_coeff4_map = switch_coeff4_map.ravel().tolist() 
    switch_coeff5_map = switch_coeff5_map.ravel().tolist() 
    manning_coeff_map = manning_coeff_map.ravel().tolist() 
    cutoff_r1_map = cutoff_r1_map.ravel().tolist() 
    cutoff_r2_map = cutoff_r2_map.ravel().tolist()

    

    assert switch_coeff[0] == 1
    assert np.sum(np.array(switch_coeff)) == 0
    
    
    # Spell out the switch function;
    elec = mm.CustomNonbondedForce(f'''energy;
           energy=manning_coeff*q1*q2*{alpha_value}*exp(-r/ldby)*switch/(dielectric*r);
           switch = (switch_coeff0 + switch_coeff1*((r-cutoff1_value)/(cutoff2_value-cutoff1_value)) + switch_coeff2*((r-cutoff1_value)/(cutoff2_value-cutoff1_value))^2 + switch_coeff3*((r-cutoff1_value)/(cutoff2_value-cutoff1_value))^3 + switch_coeff4*((r-cutoff1_value)/(cutoff2_value-cutoff1_value))^4 + switch_coeff5*((r-cutoff1_value)/(cutoff2_value-cutoff1_value))^5) * step(r-cutoff1_value)*step(cutoff2_value-r)+step(cutoff1_value-r);
           switch_coeff0={switch_coeff[0]};
           switch_coeff1=switch_coeff1_map(atom_type1, atom_type2);
           switch_coeff2=switch_coeff2_map(atom_type1, atom_type2);
           switch_coeff3=switch_coeff3_map(atom_type1, atom_type2);
           switch_coeff4=switch_coeff4_map(atom_type1, atom_type2);
           switch_coeff5=switch_coeff5_map(atom_type1, atom_type2);
           manning_coeff=manning_coeff_map(atom_type1, atom_type2);
           cutoff1_value=cutoff_r1_map(atom_type1, atom_type2);
           cutoff2_value=cutoff_r2_map(atom_type1, atom_type2);
           ldby=(dielectric*{gamma_value})^0.5;
           dielectric={A}+{B}/(1+{kappa}*exp(-{zeta}*{B}*r));
           ''')

    # Add the tabulated potential for p-p, p-n and n-n interactions;
    elec.addTabulatedFunction('switch_coeff1_map', mm.Discrete2DFunction(n_atom_types, n_atom_types, switch_coeff1_map))
    elec.addTabulatedFunction('switch_coeff2_map', mm.Discrete2DFunction(n_atom_types, n_atom_types, switch_coeff2_map))
    elec.addTabulatedFunction('switch_coeff3_map', mm.Discrete2DFunction(n_atom_types, n_atom_types, switch_coeff3_map))
    elec.addTabulatedFunction('switch_coeff4_map', mm.Discrete2DFunction(n_atom_types, n_atom_types, switch_coeff4_map))
    elec.addTabulatedFunction('switch_coeff5_map', mm.Discrete2DFunction(n_atom_types, n_atom_types, switch_coeff5_map))
    elec.addTabulatedFunction('manning_coeff_map', mm.Discrete2DFunction(n_atom_types, n_atom_types, manning_coeff_map))
    elec.addTabulatedFunction('cutoff_r1_map', mm.Discrete2DFunction(n_atom_types, n_atom_types, cutoff_r1_map))
    elec.addTabulatedFunction('cutoff_r2_map', mm.Discrete2DFunction(n_atom_types, n_atom_types, cutoff_r2_map))

    elec.addPerParticleParameter('atom_type')
    elec.addPerParticleParameter('q')

    # add atom type
    for i, row in mol.atoms.iterrows():
        resname_i = row['resname']
        name_i = row['name']
        q = row['charge']
        if (resname_i in _amino_acids):
            elec.addParticle([0, q])
        else:
            elec.addParticle([1, q])

    for i, row in mol.exclusions.iterrows():
        elec.addExclusion(int(row['a1']), int(row['a2']))
    if mol.use_pbc:
        elec.setNonbondedMethod(elec.CutoffPeriodic)
    else:
        elec.setNonbondedMethod(elec.CutoffNonPeriodic)
    elec.setCutoffDistance(4.0)
    elec.setForceGroup(force_group)
    return elec

