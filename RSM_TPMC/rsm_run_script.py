#!/usr/bin/env python3

#GAF: 2023

import math
import os
import sys
import numpy as np
import importlib
import glob
import h5py
import json
import logging
import argparse

from lhsu import lhsu
from rotate_stl import rotate_stl
from regression_model import regression_model
import argparse
import trimesh

#lhsu = imp.load_source("lhsu", "./Simulation/LHS/lhsu.py")

D2R = math.pi/180.0 #Degree to radians

def CWT(inputdir,meshdir,filename):       #Function created by Logan Sheridan at WVU


####################### Analyze STL #######################
    meshpath = os.path.join(meshdir, filename)
    mesh = trimesh.load(meshpath)

    if mesh.is_watertight:
        print('The Mesh is Watertight! \n Continuing with TPMC...')
    else:
        print('The Mesh is NOT Watertight! \n Terminating program...')
        exit()



###################### CLEAN-UP TPM DIAGNOSTIC FILES #######################
def tpm_cleanup(tpmdir):
    diagfiles = glob.glob(tpmdir+"/tpmout*")

    for i in diagfiles:
        print("Removing : %s" % (tpmdir+os.sep+i))
        os.remove(tpmdir+os.sep+i)

    #if not diagfiles == []:
    #    cmd = "rm "+tpmdir+"/tpmout*"
    #    os.system(cmd)

#    ensfile = glob.glob(tpmdir+"/tpm.ensemble.h5")
#    if not ensfile == []:
#        cmd = "rm "+tpmdir+"/tpm.ensemble.h5"
#        os.system(cmd)





########################### READ INPUT PARAMETERS ############################

def read_input(inputdir,meshdir,compdir, D2R, rsmname):
    #This function reads in a json file to a python dictionary.
    #Formatting for the json file is important. However, white space and input order is not important


    #Open the input file
    inputpath = os.path.join(inputdir, "Simulation.json")
    with open(inputpath) as rf:
        md = json.load(rf)

    NENS = md['Number of Ensemble Points']
    NPROCS = md['Number of Processors to be used']
    GSI_MODEL = md['Gas Surface Interaction Model (1=DRIA,2=CLL)']
    Rotation = md['Component Rotation (1=No,2=Yes)']
    ComponentRotation = md['Component Rotation Min/Max']


    #Number of leading variables for GSI model
    if GSI_MODEL == 0:
        k = 6
    elif GSI_MODEL == 1:
        k = 6
    elif GSI_MODEL == 2:
        k = 7
    else:
        print('Please enter valid GSI_MODEL flag in Simulation.json')
        sys.exit(1)

    RSMNAME = rsmname

    parentpath = os.path.join(inputdir, "STL_Rotation_Inputs/parent.txt")
    parents = open(parentpath,"r")
    parentnames = parents.read().split()
    num_components = len(parentnames)
    parents.close()

    if Rotation ==1:
        NVAR = k
    elif Rotation == 2:
        NVAR = k + num_components  #Number of variables rely on number of components
    else:
        print('Please enter valid component rotation flag in Simulation.json')
        sys.exit(1)


    #Initialize min and max arrays
    x = np.zeros((NVAR,2))
    xmin = np.zeros(NVAR)
    xmax = np.zeros(NVAR)
    cr = np.zeros([num_components,2])

    x[0] = md['Magnitude of Bulk Velocity']
    x[1] = md['Yaw Angle Orientation']
    x[2] = md['Pitch Angle Orientation']
    x[3] = md['Satellite Surface Temperature']
    x[4] = md['Atmospheric Translational Temperature']


    if GSI_MODEL == 0:
        x[5] = md['Specular Fraction']
    if GSI_MODEL == 1:
        x[5] = md['Energy Accommodation Coefficient']
    if GSI_MODEL == 2:
        x[5] = md['Normal Energy Accommodation Coefficient']
        x[6] = md['Tangential Momentum Accommodation Coefficient']

    if Rotation ==1:  #Move stl file to tpmc directory if no rotation
        originalfile = os.path.join(compdir,RSMNAME)
        outpath = os.path.join(meshdir, RSMNAME)
        cmd = 'cp '+originalfile+' "'+outpath+'"'
        os.system(cmd)

    if Rotation == 2:
      x[k:,0] = ComponentRotation[::2]
      x[k:,1] = ComponentRotation[1::2]




    xmin = x[:,0]
    xmax = x[:,1]


    #Convert degrees to radians
    xmin[1] *= D2R
    xmax[1] *= D2R
    xmin[2] *= D2R
    xmax[2] *= D2R


    return xmin, xmax, GSI_MODEL, NENS, NPROCS, RSMNAME, Rotation

##################### MODIFY INPUT PARAMETERS for tpm.c #######################
def modify_input(RSMNAME,GSI_MODEL,Rotation,tpmdir, species, D2R):
    #Open the temp variable file
    #This file is a way of passing variables to the tpm.c code
    #The only four variables need to pass are the object, the GSI flag,the Rotationflag and the species mole faction


    temppath = os.path.join(tpmdir, "temp_variables.txt")
    fin = open(temppath, "w")
    #Change mole fraction array
    data = "%lf %lf %lf %lf %lf %lf" % (species[0], species[1], species[2],
                                                species[3], species[4], species[5])

    line1 = "Object Name                                               # " + RSMNAME +"\n"
    line2 = "Gas Surface Interaction Model                             # " + str(GSI_MODEL)+ "\n"
    line3 = "Species Mole Fractions : X = [O, O2, N, N2, He, H]	   	# " + data +"\n"
    line4 = "Component Rotation Flag                                   # " + str(Rotation)
    L = [line1,line2,line3,line4]

    fin.writelines(L)
    fin.close()





###################### GENERATE LHS ENSEMBLE  ##########################
def lhs_ensemble(tpmdir, xmin, xmax, NENS, GSI_MODEL):

    #CREATE LATIN HYPERCUBE SAMPLE
    LHS = lhsu(xmin, xmax, NENS)

    #OPEN ENSEMBLE FILE
    ensemblepath = os.path.join(tpmdir, "tpm.ensemble")
    #open(ensemblepath, "w")

    #DEFINE HEADER
    if GSI_MODEL == 0:
        header = "Umag [m/s]     theta [radians]     phi [radians]       Ts [K]     Ta [K]     epsilon     (Remaining Columns are Rotations of n-components) "
    if GSI_MODEL == 1:
        header = "Umag [m/s]     theta [radians]     phi [radians]       Ts [K]     Ta [K]     alpha     (Remaining Columns are Rotations of n-components) "
    if GSI_MODEL == 2:
        header = "Umag [m/s]     theta [radians]     phi [radians]       Ts [K]     Ta [K]     alphan     sigmat     (Remaining Columns are Rotations of n-components) "
    #WRITE ENSEMBLES
    np.savetxt(ensemblepath, LHS, header=header)

    #Save to hdf5 file, this will help C code allocate memory efficiently for when rotation is preformed
    data=np.loadtxt(ensemblepath)
    with h5py.File(tpmdir+os.sep+"tpm.ensemble.h5", "w") as f5:
        f5.create_dataset("tpm", data=data)

    f5.close()


##################### GENERATE Deflection File  ########################

def deflectionfile(tpmdir,inputdir,outputdir):    #function created by Logan Sheridan at WVU
    #Get number of components from parent.txt file
    parentpath = os.path.join(inputdir, "STL_Rotation_Inputs/parent.txt")
    parents = open(parentpath,"r")
    parentnames = parents.read().split()
    num_components = len(parentnames)
    parents.close()

    #load in tpm.ensemble.h5 to get deflection values created by lhsu
    tpmpath = os.path.join(tpmdir,"tpm.ensemble.h5")
    hf = h5py.File(tpmpath, 'r')
    data = hf[('tpm')]
    deflection = data[:,-num_components:]

    #Create new deflection file

    deflecpath = os.path.join(outputdir,"deflections.txt")
    if os.path.exists(deflecpath):  #in case the NENS is less than last time
        os.remove(deflecpath)

    np.savetxt(deflecpath,deflection,fmt="%.3f")
    #restrict to 3 decimals, for the sake of filename length from rotation


######################## RUN TPMC ENSEMBLE #############################
def run_tpmc(path4tpm, NPROCS, speciesnames, ispec, rtype, outfile, RSMNAME):
    #RUN THE CODE
    print("Starting Simulation for "+speciesnames[ispec]+" "+rtype+" set\n")
    cmd = "mpiexec -n %d %s" % (NPROCS, path4tpm)
    logging.debug("Running tpm as: %s" % cmd)
    os.system(cmd)
    # make a fake output file
    # with open("tempfiles/Cdout.dat", "w") as f:
    #     f.write("7.337475e+03 1.364221e+03 1.344518e+03 8.201282e-01 8.968926e-02 3.134909e+00 9.881161e-01 2.551345e+00\n" 
    #             + "7.816881e+03 3.893836e+02 1.044514e+03 3.101161e-01 8.918926e-01 2.864616e+00 -1.496969e+00 4.034787e+00")

    #COPY THE OUTPUT TO A NEW FILE
    print("Copying output data\n")
    # outname = RSMNAME+"_"+speciesnames[ispec]+"_"+rtype+"_set.dat"
    # outpath = os.path.join(outdir, outname)
    cmd = 'cp tempfiles/Cdout.dat "'+outfile+'"'
    os.system(cmd)

    print("Copying area output data\n")
    areaoutname = RSMNAME+"_"+speciesnames[ispec]+"_"+rtype+"_area.dat"
    areaoutpath = os.path.join('Outputs/Projected_Area/', areaoutname)
    cmd = 'cp Outputs/Projected_Area/Aout.dat "'+areaoutpath+'"'
    os.system(cmd)



################ ADD ROTATION INFO TO TEST/TRAINING DATA #####################
def rotation_data(rtype,GSI_MODEL,NENS):  #function created by Logan Sheridan at WVU
    """
    This function is to save the rotation information from the LHS into the
    training and test data


    Parameters
    ----------
    rtype : training or test data.
    GSI_MODEL: gsi model being used
    NENS : number of ensemble points
    Returns
    -------
    None.

    """

    species = ['H','He','O','O2','N','N2']


    if GSI_MODEL == 1:
        gsi_col = 6
    elif GSI_MODEL ==2:
        gsi_col = 7


    for n in range(len(species)):


        values=[]
        areafile = "Outputs/Projected_Area/"+RSMNAME+"_"+species[n]+"_"+rtype+"_area.dat"
        fcount = open(areafile,'r')
        line_count =0
        for i in fcount:
            if i != "\n":
                line_count += 1
        fcount.close()

        f = open(areafile,'r')
        j=0
        for line in f:

            data = line.split()
            floats = []
            for elem in data:
                try:
                    floats.append(float(elem))
                except ValueError:
                    pass
            if j == 0:
                columns = len(floats)
                values = np.zeros([line_count,columns])
            values[j,:] = np.array(floats)
            j+= 1

        f.close()


        points = values[:,1:]


        ROTATION_VALUES = points[:,2:]

        columns = gsi_col + len(ROTATION_VALUES[0]) +1
        new_data = np.ones([NENS,columns])

        if rtype == "training":
            ftype ="Training"
        elif rtype =="test":
            ftype="Test"


        tpmcdata = np.loadtxt("Outputs/RSM_Regression_Models/data/"+ftype+" Set/"+RSMNAME+"_"+species[n]+"_"+rtype+"_set.dat")

        lhs = tpmcdata[:,:-1] #all columns except last
        cd = tpmcdata[:,-1] #last column

        new_data[:,:gsi_col] = lhs
        new_data[:,(gsi_col):-1] = ROTATION_VALUES
        new_data[:,-1] =cd

        #overwrite old data to include the rotations in test and training data
        np.savetxt("Outputs/RSM_Regression_Models/data/"+ftype+" Set/"+RSMNAME+"_"+species[n]+"_"+rtype+"_set.dat",new_data)



###################### MAKE FILE WITH ALL AREA INFO ###########################
def total_areafile(RSMNAME):     #function created by Logan Sheridan at WVU
    outfilename = 'Outputs/Projected_Area/Area_Total.dat'
    with open(outfilename, 'w') as outfile:
        for filename in glob.glob('Outputs/Projected_Area/'+RSMNAME+'*.dat'):
            if filename == outfilename:
                # don't want to copy the output into the output
                continue
            with open(filename, 'r') as readfile:
                outfile.write(readfile.read())


############### LOOP OVER TEST/TRAINING AND MOLE FRACTIONS ##############
###################### RUN ROTATION OF STL FILES ########################
def tpmc_loop(path4tpm, speciesnames, NPROCS, NENS,GSI_MODEL,Rotation, xmin,xmax, regdir, tpmdir,basedir, RSMNAME, outfile):

    ####################### CONSTANTS #######################
    D2R = (math.pi/180.0)   #DEGREES TO RADIANS CONVERSION
    # NSPECIES = 6            #NUMBER OF SPECIES
    # speciesnames = ["O", "O2", "N", "N2", "He", "H"]
    # species = np.zeros(NSPECIES)

    NSPECIES = len(speciesnames)
    species = np.zeros(NSPECIES)
    species = [1 if "O" in speciesnames else 0,
               1 if "O2" in speciesnames else 0,
               1 if "N" in speciesnames else 0,
               1 if "N2" in speciesnames else 0,
               1 if "He" in speciesnames else 0,
               1 if "H" in speciesnames else 0]
    

    #os.chdir(tpmdir)
    if os.path.exists("tempfiles/tpm.ensemble.h5"):
        os.remove("tempfiles/tpm.ensemble.h5")
    # if os.path.exists("tempfiles/tpm.ensemble"):
    #     os.remove("tempfiles/tpm.ensemble")
    outputdir = os.path.join(basedir,"Outputs")
    #Create LHS ensemble
    lhs_ensemble(tpmdir, xmin, xmax, NENS, GSI_MODEL)

    if Rotation ==2:
    #Create Deflection file
        deflectionfile(tpmdir,inputdir,outputdir) # deflection file used by rotate_stl.py
    ## Run STL Rotation Script
    #Note: this script uses the Simulation.json "Object Name" to create the new body
        runRotation(basedir)                    #Rotate Components of satellite

    #os.chdir(tpmdir)
    if os.path.exists("tpm.ensemble.h5"):
        os.remove("tpm.ensemble.h5")

    # for idx in range(2):  #Loop over to create test then training data
    for ispec in range(NSPECIES):   # Loop over each species of mole fraction for tpmc
        rtype = "training"
        # outdir = os.path.join(outputdir, "RSM_Regression_Models/data/Training Set/")

        # #Create directory if it doesn't exist
        # if not os.path.exists(outdir):
        #         os.makedirs(outdir)

        #Define Species Array
        # for i in range(NSPECIES):
        #     species[i] = 0.0
        #     if i == ispec:
        #         species[i] = 1.0

        #Modify Species Mole Fraction for tpmc simulation
        modify_input(RSMNAME,GSI_MODEL,Rotation, tpmdir, species, D2R)

        #Run TPMC code
        run_tpmc(path4tpm, NPROCS, speciesnames, ispec, rtype, outfile, RSMNAME)

#If rotation is being performed, then the test and training data needs to include all componente rotation values
    if Rotation ==2:
                rotation_data(rtype,GSI_MODEL,NENS)

    total_areafile(RSMNAME)

#################### ROTATE COMPONENTS OF SATELLITE# #########################
def runRotation(rotationpath):
    print("Rotation Path:" + rotationpath)
    print(os.getcwd())
    rotate_stl(rotationpath)
    #exec(open("./rotate_stl.py").read())

################### RUN THE REGRESSION MODEL FITTING #########################
def run_reg(basedir):
    #Clean-up parent directory
    cmd = "rm ./tempfiles/Cdout*.dat"
    os.system(cmd)
    regression_model(basedir)

def logo():
	logo="""
 ___                                             ___                ___                                     _       _
|  _ \                                          (  _ \            / ___)                   / \_/ \         ( )     (_ )
| (_) )  __   ___ _ _     _    ___   ___   __   | (_(_)_   _ _ __| (__    _ _   ___   __   |     |  _     _| |  __  | |
|    / / __ \  __)  _ \ / _ \/  _  \  __)/ __ \  \__ \( ) ( )  __)  __) / _  )/ ___)/ __ \ | (_) |/ _ \ / _  |/ __ \| |
| |\ \(  ___/__  \ (_) ) (_) ) ( ) |__  \  ___/ ( )_) | (_) | |  | |   ( (_| | (___(  ___/ | | | | (_) ) (_| |  ___/| |
(_) (_)\____)____/  __/ \___/(_) (_)____/\____)  \____)\___/(_)  (_)    \__ _)\____)\____) (_) (_)\___/ \__ _)\____)___)
                 | |
                 (_)                                                                                                    """

	print(logo)

##########################################################################
##################### RUN AUTOMATED RSM GENERATION #######################
##########################################################################


if __name__ == '__main__':
    
    def list_of_strings(arg):
        return arg.split(',')

    logging.basicConfig(level=logging.DEBUG)
    logo()

    print("Surface Model Simulation")

    parser = argparse.ArgumentParser(description='Response Surface Model')
    parser.add_argument('--tpm', metavar='PATH', type=str,
                            help='Path for executable tpm')
    parser.add_argument('--input', type=str, required=True, help='Object Name')
    parser.add_argument('--output', type=str, required=True, help='Output file')
    parser.add_argument('--species', type=list_of_strings,
                        help='Species names seperated by comma', 
                        default=["O", "O2", "N", "N2", "He", "H"])

    args = parser.parse_args()

    if args.tpm is None:
        parser.print_help()
        sys.exit(1)

    # if not os.path.isfile(args.tpm):
        # raise ValueError("Could not find %s", args.tpm)

    #################### Directories ########################
    print("Checking folders...")
    basedir = os.getcwd()
    topdir = os.path.join(basedir,'.')
    inputdir = os.path.join(topdir, "Inputs")
    rotationdir = os.path.join(topdir,"RSM_TPMC")
    tpmdir  = os.path.join(basedir, "tempfiles")
    regdir = os.path.join(topdir, "RSM_TPMC")
    compdir = os.path.join(inputdir,"STL_Files")
    meshdir = os.path.join(tpmdir, "Mesh_Files")
    outputdir = os.path.join(basedir, "Outputs")

    if not os.path.isdir(tpmdir):
        os.mkdir(tpmdir)

    if not os.path.isdir(meshdir):
        os.mkdir(meshdir)

    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)

    for i in [ basedir,topdir, inputdir, rotationdir, tpmdir, regdir, compdir, meshdir ]:
        if not os.path.isdir(i):
            print("Folder %s could not be found" % i)
            sys.exit(1)

    ####################### INPUTS ##########################
    print("Reading input files...")
    rsm_name = os.path.basename(args.input)

    xmin, xmax, GSI_MODEL, NENS, NPROCS, RSMNAME, Rotation = read_input(inputdir,meshdir,compdir, D2R, rsm_name)
    outfile = args.output

    ################### START CODE ########################

    #Check the watertightness of STL if there is no rotation being performed
    if Rotation == 1:
        CWT(inputdir,meshdir,RSMNAME)
    #Cleanup TPMC diagnostic files (If it crashed last time)
    tpm_cleanup(tpmdir)

    #Run TPMC test and training sets -> Includes reading input, and rotating stl files
    print("TPMC test and training sets...")
    tpmc_loop(args.tpm, args.species, NPROCS, NENS,GSI_MODEL,Rotation,xmin,xmax, regdir, tpmdir,basedir, RSMNAME, outfile)

    #Run Regression to fit data
    # print("Regression...")
    # run_reg(basedir)

    #Clean up TPMC diagnostic files
    tpm_cleanup(tpmdir)

