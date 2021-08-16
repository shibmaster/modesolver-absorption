#!/bin/python3

#### INITIALIZATION ####

#!/bin/python3
#Getting the Libraries
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import numpy as np
from numpy.random import multivariate_normal
import scipy.constants as cnt
from sympy import Eq, var, solve, Symbol
import scipy.interpolate as scinter
from matplotlib.collections import LineCollection


#Global Constants
Pi=cnt.pi
n0=3.645
C0=cnt.c   #Speed of Light in Vacuuo


#Import DATA of REFRACTIVE INDEX
n1r=np.loadtxt("siliconR.txt", skiprows=1) 
n1i=np.loadtxt("siliconI.txt", skiprows=1) 
#Creating interpolated data functions
n1ri=scinter.interp1d(n1r[:,0],n1r[:,1], fill_value='extrapolate')
n1ii=scinter.interp1d(n1i[:,0],n1i[:,1], fill_value='extrapolate')

#Import DATA for the SOLAR SPECTRUM AM1.5
am15=np.loadtxt("AM15G.dat",skiprows=1)#, usecols= (1,2,3,)) # this ARRAY: Lambda (nm), Intensity
am15flux=[]
#Creating a dictionary for the solar flux per lambda in nanometers
amdict={}
for i in am15:
	L=i[0]/1e9              #Wavelength is in column one, in nanometers
	Lmu=round(i[0]/1000,4)  #We save WL also in microns 
	Lint=int(i[0])          # As well as keep it in nm, but make sure it is int
	E=cnt.h*cnt.c/L         # Calculate the photon energy at lambda
	N=i[1]/E                # Calculate the flux at lambda from intensity/energy
	am15flux.append([Lmu,4,N])
	amdict[Lint]=N
am15flux=np.asarray(am15flux)
am15fluxi=scinter.interp1d(am15flux[:,0],am15flux[:,2], fill_value='extrapolate')


#IMPORT THE MODESOLVER OUTPUT
def importModes(name,delim=','): #If this throws an error, check delimiter and skiprows in the file!
    global modes,lambdastart,lambdastep,lambdamax,lambdasteps, lambar
    modes=np.loadtxt(name, delimiter=delim, skiprows=1)
    lambdastart=modes[0,1]
    lambdastep=.0001
    lambdamax=lambdastart
    lambdasteps=0
    i=1
    while lambdamax < modes[i,1]:
        lambdamax=modes[i,1]
        i+=1
    lambdasteps=i
    Lambs=[.4,1.051,lambdastep]
    lambar=[]
    LL,lamb=0.0,1.0
    i=0
    while lamb>LL:
        lambar.append(modes[i,1])
        LL=modes[i,1]
        i+=1
        lamb=modes[i,1]
        lambdastep=round(max(lambdastep,lamb-LL),3)
    print("Found the following mode library characteristics:")
    print("Lambda start/max, step is, total steps",lambdastart,"/",lambdamax, lambdastep, lambdasteps)
    
    
    
    
##### SUBROUTINES ######
def extractKL(k1,k2,lamb):   #Filter the mode data for points in the wavelength range with k_1<k_m<k_2
    ar1 = modes[np.argsort(modes[:, 2])]
    #print(array,'\n\n')              
    newar=[]
    for line in ar1:
        if line[2]<=k2 and line[2]>=k1:
            newar.append(line)
        if line[2]>k2:
            break
    if len(newar)>0:
        ar2=np.asarray(newar)        
        ar22 = ar2[np.argsort(ar2[:, 1])]
        newar=[]
        for line in ar22:
            if abs(line[1]-lamb)<lambdastep:
                newar.append(line)
        if len(newar)>0:
            ar3=np.asarray(newar)
            ar33=ar3[np.argsort(ar3[:, 0])]
            return ar33
        else:
            return 0
    else:
        return 0
    
    #Equation 7	Version as outlined above
def eq7a(lamb,k1,k2):		#Equation 7
    array=extractKL(k1,k2,lamb)
    #print(array)
    k0=2*Pi/lamb
    #if len(array)==1: #If no modes exist / are found
    if isinstance((array), int):    
        return 0.0
        #print("empty array")
    else:
        res=0.0
        for line in array:
            km=line[2]      #Propagation constant is in column 3
            alpha=2*line[3] #Extinction coefficient in column 4
            nm=km/k0
            res+= 2*Pi*alpha*km/(k2**2-k1**2)
        return res		

    
    #Equation 7	With N_m evaluated through Femius' equation, compare with Thesis sec. (2.2)
def eq7b(lamb,k1,k2):
    array=extractKL(k1,k2,lamb)
    #print(array)
    k0=2*Pi/lamb
    #if len(array)==1: #If no modes exist / are found
    if isinstance((array), int):    
        return 0.0
        #print("empty array")
    else:
        res=0.0
        for line in array:
            km=line[2]
            alpha=2*line[3]
            alpha=averab(line[1])
            nm=km/k0
            res+= 2*Pi**2*alpha*km*nm/(k2**2-k1**2)
        return res		

def eq8a(alpha,lamb): #Equation 8, without enhancement (cp. with green2002:Analytical solutions)
    n=n1ri(lamb)
    return (1 - np.exp(-alpha) )/ (1 - (1 - 1/n**2)*np.exp(-alpha)  )


def eq8b(alpha,lamb): #Equation 8    with optical path enhancement factor f_p = 4
    n=n1ri(lamb)
    return (1 - np.exp(-4*alpha) )/ (1 - (1 - 1/n**2)*np.exp(-4*alpha)  )

def IA(k1,k2,eq7,eq8):       #Integration of EQ8 over Wavelengths weighed by the AM1.5 Spectrum
    if k1<k2:
        xdata=np.arange(.4,1.051,lambdastep)
        #Ls=np.arange(Lstart,Lend+Ld,Ld)
        A=[eq8(eq7(x,k1,k2),x) for x in xdata]
        A=np.asarray(A)
        tot=np.trapz(am15fluxi(xdata)*A,xdata)
        norm=np.trapz(am15fluxi(xdata),xdata)        
        return tot/norm
    else:
        return 0.0

#Some additional functions for individual inspection

def lambertAb(lamb,thick):  #Single-Pass Absorption with Lambert-Beer law
    return (1-np.exp(-2*n1ii(lamb)*thick*2*cnt.pi/lamb))

#Calculate the absorption coefficient for a certain free space lambda, a=2*im(n)*k0
def averab(lamb):
	alpha=2*2*Pi*n1ii(lamb)/(lamb)
	return alpha

def gammai(lamb):    #Calculate the intrinsic loss rate gamma_i for certain lambda
    n=n1ri(lamb)
    c=C0
    alpha=averab(lamb)*10**6     #For meters since averab is in 1/um
    return alpha*c/n
    
    
    
    
    
    
    
    
    
##### PLOT FUNTIONS    #####
#First a quick data-export function, to save the long calculated data
def Fileprint(name,array):
    file=open("{}.dat".format(name), 'w')
    #file.write('')
    file.close()
    file=open("{}.dat".format(name), 'a')
    sr=''
    for line in range(len(array)):
        sr=''
        for i in range(len(array[line])):
            sr+=str(array[line,i])
            sr+=','
        sr=sr[:-1]
        sr+='\n'
        file.write(sr)
    file.close()

#Plotfunction of Absorption for some k1,k2
def plAbsorption(k1,k2,eq7,eq8,name=''):
    xdat=np.asarray(np.arange(lambdastart,lambdamax+lambdastep,lambdastep))
    ydat=np.asarray([ eq8(eq7(x,k1,k2),x) for x in xdat])
    ydat2=np.asarray([ lambertAb(x,1) for x in xdat])
    ex=[[xdat[i],ydat[i]] for i in range(len(xdat))]
    np.savetxt("Absorpv13{}-{}-{}.dat".format(name,k1,k2), np.asarray(ex), delimiter=',')
    fig, ax = plt.subplots(1,1)
    #ax.plot(xdat,ydat, 'g--', label='EQ8,k1={}, k2={}'.format(k1,k2))
    ax.scatter(xdat,ydat,s=1, label='EQ8,k1={}, k2={}'.format(k1,k2))
    ax.plot(xdat,ydat2, 'b--', label='Single-Pass')
    ax.set_xlabel('Wavelength / microns')
    ax.set_ylabel('Absorption')
    ax.grid()
    ax.legend()
    fig.savefig("Absorption{}-{}-{}.pdf".format(name,k1,k2), bbox_inches='tight')
    
# PLOT K1 vs K2, takes LONG time if delta is small and kmax-kmin is large!
def plk1k2(kmin,kmax,eq7,eq8,name='Test'):			#Plotfunction of dispersion/absorption for kx vs. k0 for thickness h (um), name output name
    delta=1
    ks=np.arange(kmin,kmax,delta)
    abbs=[]
    #abbs=[[IAp(p,j,eq7,eq8,*Lambs) for j in ks] for p in ks]  # Use this line and comment out loop below to suppress progress output.
    print("current k:",end="")
    for j in ks:
        print(" ,",j,end="")
        for p in ks:
            abbs.append([j,p,IA(j,p,eq7,eq8)])
    abbs=np.asarray(abbs)  
    Fileprint("IA{}-{}-{}".format(name,kmin,kmax), abbs)
    fig, ax = plt.subplots(1,1)		
    ax.scatter(abbs[:,0],abbs[:,1], marker=',',s=700*delta**2, c=abbs[:,2], vmin=0,vmax=1)
    ax.set_xlabel('K1 / $\mu$m$^{-1}$')
    ax.set_ylabel('K2 / $\mu$m$^{-1}$')
    ax.set_title("Spectrally averaged Integrated Absorption")
    fig.colorbar(ax.collections[0],ax=ax,label='Absorption')
    ax.grid()
    #ax.legend()
    fig.savefig("K1vsK2-{}-Ks{}-{}.pdf".format(name,kmin,kmax), bbox_inches='tight') #,norm=mpl.colors.Normalize(vmin=0.3, vmax=.5)) 



#Plotfunction for AM spectrum and absorbed fraction               
def plSpectrum(k1,k2,eq7,eq8):			#Plotfunction for AM spectrum and absorbed fraction
    fig, ax = plt.subplots(1,1)
    x=np.asarray(am15[:,0]/1000)
    y=np.asarray(am15[:,1])
    #y2=[absorp(abscoeff(o,k1,k2)) for o in x]
    y3=np.asarray([eq8(eq7(o,k1,k2),o) for o in x])
    y4=np.asarray([(lambertAb(u,1)) for u in x])
    ax.plot(x,y, '-', color='b',	label='AM1.5 Spectrum')
    ax.plot(x,y4*y, '--', color='y',	label='Single Pass Lambert')
    ax.plot(x,y3*y, '--', color='grey',	label='EQ8,{0:.1f}-{1:.1f} IA={2:.3f}'.format(k1,k2,IA(k1,k2,eq7,eq8)))
    #ax.plot(n1i[:,0],n1i[:,1], 'g--', color='g',	label='im')
    ax.set_xlabel('Wavelength / $\mu$m')
    ax.set_ylabel('(Absorbed) Irradiation / W/m**2/micron')
    ax.set_xlim(.400,1.200)
    ax.grid()
    ax.legend()
    fig.savefig("AM1.5and-k1k2-{}-{}.pdf".format(k1,k2), bbox_inches='tight')

    
#Plotfunction for refractive index
def plRef():			#Plotfunction for refractive index
    fig, ax = plt.subplots(1,1)
    ax.plot(n1r[:,0],n1r[:,1], 'g--', color='b',	label='real')
    ax.plot(n1i[:,0],n1i[:,1], 'g--', color='g',	label='im')
    ax.set_xlabel('Wavelength / um')
    ax.set_ylabel('Absorption')
    ax.grid()   
    ax.set_yscale('log')
    ax.legend()
    fig.savefig("RefractiveIndex.pdf", bbox_inches='tight')









#### EXAMPLES ####

# START BY IMPORTING THE MODE DATA, OR GENERATE IT WITH MODESOLVER CLASS #

#from MSclass13 import *
#M=Modedata(Layers=['air','silicon','air'], Thickness=1, Lambdas=[0.4,1.1,0.05])
#M.Precision=4
#M.generate()
#M.Export('ASiA-modes-5nm')
#M.Import('ASiA-modes-5nm')
#M.plDisp('ASIA-modes')

### IMPORT MODES ###
#importModes('ASiA-modes-5nm.dat')


### CALCULATE INTEGRATED ABSORPTION FOR K1, K2 ###
#print("Integrated Absorption K1=10, K2=20:,",round(100*IA(10,20,eq7a,eq8a),1),'%')
