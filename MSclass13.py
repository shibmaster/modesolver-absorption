#GET NECESSARY LIBRARIES
import sys
import numpy as np
import scipy.constants as cnt
import scipy.interpolate as scinter   
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib.ticker import FixedLocator,AutoMinorLocator

class Modedata:

#### GENERAL FUNCTIONS OF THE OBJECT #####
    def __init__(self,Lambdas=[0.4,1.1,.01],Layers=["air","silicon","air"],Thickness=1,Precision=10,smk=1):
        self.Lsta=Lambdas[0]
        self.Lend=Lambdas[1]
        self.Ldel=Lambdas[2]
        #self.Lsps=int((Lambdas[1]-Lambdas[0])/Lambdas[2])
        self.Layers=Layers
        self.Thickness=Thickness
        self.Precision=Precision
        self.beta=[]
        self.all=[]
        self.mindex=[]
        self.smk=smk
        self.n1Ri, self.n1Ii=self.importRefractive(Layers[1])
        self.n2Ri, self.n2Ii=self.importRefractive(Layers[0])
        self.n3Ri, self.n3Ii=self.importRefractive(Layers[2])
        
    def __str__(self):
        if len(self.beta) <1:
            print('This modedata has not been generated. Generate with OBJ.generate(), or type OBJ.help() for more instructions')
        else:
            print('The modedata has been generated.')
        return 'The current characteristics are:\n Lambda min: {} um, Lambda max: {} um, Lambda step: {} um\n Thickness: {} um, \n Layers: {}, \n Precision: {} decimal places.'.format(self.Lsta,self.Lend,self.Ldel,self.Thickness,self.Layers,self.Precision)
    
    def generate(self,Mmin=0,Mmax=-1):         #This function actually generates the mode data of the object
        e,m,modes,idx=[],[],[],[]
        if Mmax == -1:
            Mmax=self.mxmo()                #If all modes are to be generated, find the highest mode number
        print("Generating data for {} modes, current mode:".format(Mmax-Mmin))
        idx.append(0)                       #idx will become OBJ.mindex, which is an array that holds the line numbers where new modes begin
        for m in range(Mmin,Mmax):          #This loop will now calculate all modes and store them into OBJ.beta afterwards
            print("{}, ".format(m), end="")
            e=self.FindBeta('E',m)
            m=self.FindBeta('M',m)
            if len(modes) < 1:
                modes=e
            elif len(e)>1:
                modes=np.append(modes,e,axis=0)
                idx.append(len(modes))
            if len(m)>1:
                modes=np.append(modes,m,axis=0)
                idx.append(len(modes))
        print("")
        self.beta=modes
        self.mindex=idx
        
    def help(self):
        print("Thanks for using a Modedata object! You can:\n" \
              "Initialize: OBJ=Modedata(Lambdas=[0.4,1.1,.01],Layers=[\"air\",\"silicon\",\"air\"],Thickness=1,Precision=10)\n\n" \
              "These parameters are stored in the object variables:\n" \
              "Lambdas=[Lsta,Lend,Ldel], to change: e.g. OBJ.Lsta=0.4, the other variables are named as above, Layers is saved as an array.\n" \
              "Parameters can be read by printing the object with print(OBJ).\n\n" \
              "To generate the data of propagation constants and absorption, call\n OBJ.generate(), which saves the data in an array under OBJ.beta.\n" \
              "This array: OBJ.beta has four columns: MODE#, Lambda, beta, alpha/2, K0.\n\n"
              "The beta array can be exported/imported using OBJ.Export(filename) or OBJ.Import(filename), respectively.\n" \
              "To learn about other internal functions, call OBJ.Fhelp(). \n To learn about plotting options, call OBJ.Phelp()." \
             )
        
    def Fhelp(self):
        print("The functions you can execute besides OBJ.help() are: \n\n" \
        "OBJ.generate(MinModeNumber,MaxModeNumber), for generating the modedata of specific modes, or all if generate() is called,\n\n" \
        "OBJ.critang(lambda), for finding the critical angle value at wavelength lambda\n\n" \
        "OBJ.mxmo(), and mxmoL(lambda), for finding the highest mode number in the range, or of a certain wavelength lambda\n\n" \
        "OBJ.Export(name), for writing the generated modedata to an output file with name: name.dat, " \
        "the format of which will have the columns: MODE#, Lambda, beta, alpha/2, K0.\n\n" \
        "Please be aware that E, and M modes will be printed right after each other and can be separated by the jump in Lambda at the same Mode#. First are E, then M modes\n\n" \
        "The indices of all modes are saved in the array OBJ.mindex, where E modes are even and M the odd indices of mindex.\n\n" \
        "OBJ.Import(name), for importing previously exported modedata. The thickness should be provided in the first line ;=(thick), or set afterwards.\n\n" \
        "OBJ.ImportLumerical(name.txt, Columns per mode, First col with beta, First col with alpha/2), for importing Lumerical modedata. The thickness should be provided after import, CHECK YOUR COLUMNS and append filename extension.\n\n" \
        "There are several plotting options, which can be listed with OBJ.plhelp()\n" \
        )

    def Phelp(self):
        print("The included plotting functions are:\n" \
            "OBJ.plDisp(plotname, minMode,maxMode) for ploting the Dispersion diagram for specific modes, or all with only (plotname)\n\n" \
            "OBJ.plRef(layernumber) for plotting the refractive index of a layer, where 1 == Core, 2 == Upper clad, 3 == Lower clad.\n\n" \
            "OBJ.plModecount(plotname) for plotting the number of modes against wavelength in the given range\n\n" \
            )     
        
#### INITIALIZATION Functions ####
    def importRefractive(self,name): # import tabulated values for n,k of material (name), e.g. 'silicon'
        ndata=name
        lambdaAR=np.arange(self.Lsta,self.Lend,self.Ldel)
        try:
            nR=np.loadtxt("{}R.txt".format(ndata), skiprows=1) #Real Part vs. lambda in microns
        except IOError:
            print("Refractive data not accessible: {}R.txt, using real value := 1".format(ndata))
            nR=np.asarray([ [i,i/i] for i in lambdaAR])
        nRi=scinter.interp1d(nR[:,0],nR[:,1], fill_value='extrapolate')
        try:
            nI=np.loadtxt("{}I.txt".format(ndata), skiprows=1) #Imag Part vs. lambda in microns
        except IOError:
            print("Refractive data not accessible: {}I.txt, using imaginary value := 0".format(ndata))
            nI=np.asarray([ [i,0/i] for i in lambdaAR])
        nIi=scinter.interp1d(nI[:,0],nI[:,1], fill_value='extrapolate')
        return nRi,nIi
    

#### Short functions for quick transformations of units and scales #####
    def omola(self,lamb):         #Omega (THz)  from wavelength in um
        return (2*cnt.pi*cnt.c/lamb/10**9)

    def laoom(self,omeg):       #Wavelength (um)  from freq. in Thz
        return (cnt.c/omeg*(2*cnt.pi)/10**9)

    def omoe(self,E):              #Frequency (Thz) from Energy in eV
        Pi,C0,h,e=cnt.pi,cnt.c,cnt.h,cnt.e
        hev=h/e
        return 2*Pi*E/hev/10**15

    def eoom(self,w):           #Energy (eV) from Freq. in Thz
        Pi,C0,h,e=cnt.pi,cnt.c,cnt.h,cnt.e
        hev=h/e
        return w*10**15*hev/2/Pi

    def kola(self,lamb):            # calculate K-vector value from wavelength in um
        Pi,C0,h,e=cnt.pi,cnt.c,cnt.h,cnt.e
        return 2*Pi/lamb

    def laok(self,k):           # calculate wavelength from K-vector value  in 1/um
        Pi,C0,h,e=cnt.pi,cnt.c,cnt.h,cnt.e
        return 2*Pi/k

    def scale(val, src, dst): #Scale the given value from the scale of src to the scale of dst.
        return ((val - src[0]) / (src[1]-src[0])) * (dst[1]-dst[0]) + dst[0]


#### MODESOLVER functions ####
    def critang(self,lamb):         # Critical angle in radians from lambda in um of interface 0: n1 to n2, or inter=1: n2 to n3
        n=self.n1Ri(lamb)       #Core refractive index
        m=max(self.n2Ri(lamb),self.n3Ri(lamb))  # claddings refractive index
        #print("Lamb, ca1, ca3",lamb,n2Ri(lamb),n3Ri(lamb),np.arcsin(n2Ri(lamb)/n),np.arcsin(n3Ri(lamb)/n))
        return np.arcsin(m/n)+1e-14

    def mxmo(self):# Maximum number of modes for thickness thi (um) of all Lambda at interface inter
        arr=[]
        #lambs=np.arange(self.Lsta,self.Lend,self.Ldel)
        #thi=self.Thickness
        #for L in lambs:
        #    ca=self.critang(L)
        #    arr.append(int(self.TXmode(L,ca,'E')[0]))
        return int(self.TXmode(self.Lsta,self.critang(self.Lsta),'E')[0]+1)

    def mxmoL(self,lamb):# Maximum number of modes for thickness thi (um) at wavelength lamb (um) and interface inter
        ca=self.critang(lamb)
        thi=self.Thickness
        return ((self.TXmode(lamb,ca,'E')[0]))


    def PhaseshiftTE(self,inter,lamb,theta): # Calculate the phase shift for a reflected TE mode at interface inter
        n=self.n1Ri(lamb)
        if inter < 1:
            m=self.n2Ri(lamb)
        else:
            m=self.n3Ri(lamb)
        root=np.sqrt( max(0, (np.sin(theta))**2 - (m/n)**2 ))
        shift=( np.arctan( root / np.cos(theta) ) )
        return shift

    def PhaseshiftTM(self,inter,lamb,theta): # Calculate the phase shift for a reflected TM mode at interface inter
        n=self.n1Ri(lamb)
        if inter < 1:
            m=self.n2Ri(lamb)
        else:
            m=self.n3Ri(lamb)
        root=np.sqrt(max(0,  ((n/m)*np.sin(theta))**2 - 1))
        shift=( np.arctan( root / (m/n) / np.cos(theta) ) )
        return shift

    def TXmode(self,lamb,theta,X):          # returns the Modenumber of X-mode (E/M) for angle theta [rad], layer thickness thi [um], Lambda lamb in [um].
        Pi,C0,h,e=cnt.pi,cnt.c,cnt.h,cnt.e
        n=float(self.n1Ri(lamb))        # the real part of the refractive index
        w=self.omola(lamb)      # the frequency corresponding to free space wavelength lambda
        k0=self.kola(lamb)      # The free space wave vector
        thi=float(self.Thickness)
        if X == 'E':        # Choose correct phase shift term for the mode type
            atan=(self.PhaseshiftTE(0,lamb,theta) + self.PhaseshiftTE(1,lamb,theta))
        elif X == 'M':
            atan=(self.PhaseshiftTM(0,lamb,theta) + self.PhaseshiftTM(1,lamb,theta))
        else: #Critical angle
            atan=0 #print("ERROR NO VALID MODE CHOSEN") 
        M = (k0 * np.cos(theta) *n *thi - atan)/Pi      # Calculate the analytical modenumber   #kx=k0 * np.sin(t) *n *h 
        N = np.sin(theta) *n                            # Calculate the effective Refr. Index
        return M,N

    def getConfinement(self,kyc,kyl,h,m):			#Calculates the confinement factor for a mode in the wave guide, kyc, kyl are decay constants in the
        def f(k,x): 					#core of the wave guide / the cladding layer, respectively
            return np.exp(-k*x)**2
        def core(k,x):
            return ((m % 2)*(np.sin(k*x)/np.sin(k*h/2))+(1+m % 2)*(np.cos(k*x)/np.cos(k*h/2)))
        def odd(k,x):
            return (np.sin(k*x)/np.sin(k*h/2))
        crange=np.arange(-h/2,h/2,h*.01)
        #core=np.trapz(f(kyc,crange),crange)
        core=abs(np.trapz(core(kyc,crange),crange))
        lrange=np.arange(h/2,10**5,h*.01)
        layer=np.trapz(f(kyl,lrange-h/2),lrange)   #(np.cos(kyc*h/2)**2
        conf=max(0,core/(2*layer+core))
        #print('Core, layer, conf', core,layer,conf)
        return conf
        
    def FindBeta(self,X,mod):                           # Calculates the effective refractive index for all Lambda supporting mode T"X":"mod"
        Pi,C0,h,e=cnt.pi,cnt.c,cnt.h,cnt.e
        Lambs=np.arange(self.Lsta,self.Lend,self.Ldel)
        arr=[]                                          # Array for returning the (Lambda, Eta, XX) tuples sorted by Mode
        sp=0                                            # Current Step count on way to precise theta T
        T=0.0
        prc=self.Precision                              # Desired Precision in digits after the decimal point, Maximum is 14
        mxp=200                                         # Cutoff threshold for approximation loops; usually no more than 70 steps are needed for 12 digit precision
        for L in Lambs:     # Go through all wavelengths in the range with resolution "Resolution"
            ca=self.critang(L)
            result=self.TXmode(L,ca,X)[0]               # Get Modenumber of critical angle 'ca' for current lambda "L"
            #print("Mode,Result min/max:",mod,L,ca,TXmode(L,thi,ca,'')[0],result)
            if result-mod>0:                            # Find if desired mode "mod" exists for current lambda "L".
                sp=0
                diff=result-mod
                u=Pi/2-1e-20                            # The angle "T" we are looking for is in the interval [l,u] = [ca,Pi/2]
                l=ca
                delta=abs(u-l)                          # The range is designated delta, which is chopped later into smaller steps
                while abs(diff) > 10**(-prc) and sp<mxp:# Repeat loops until desired precision is reached (or step count overflows)
                    sp+=1
                    delta=delta/(1.618**4)              # The step-size in current range is the fourth power of the golden ratio, 
                    T=u                                 #       but can be anything smaller than delta, really. 
                    result=self.TXmode(L,T,X)[0]            # Now we check the modenumber at a new angle, which should still be lower than "mod".
                    while result < mod and T>=l and sp<mxp: #Now we step down the guessing angle until we are higher than desired Modenumber
                        sp+=1
                        u=min(T,Pi/2-1e-20)             
                        T-=delta                    
                        result=self.TXmode(L,T,X)[0]                
                    T+=delta/2                          # The last two steps, which encircle "T", are chosen as new intervall range
                    result,N=self.TXmode(L,T,X)         # We set "T" to the center of the new interval and see if we are below "ca".
                    diff=result-mod
                    if T - l < 0:                       #If that is the case, we go in the center between current "T" and upper threshold u.
                        #print("Below Critical",X,mod,"@",int(1000*L),"nm",precision',diff)
                        T+=delta/2
                        u+=delta/2
                        sp+=mxp/10                      #Also, if that is the case, we reduce the number of allowed steps
                #print("Mode",X,mod,"Lambda",int(1000* L), 'Steps:', sp,"/",mxp)
                if abs(diff)>10**(-prc/4):  #Sometimes, the desired precision cannot be reached, if it is worse than 1 quarter of Precision, the solution does not count!
                    print("MODE",X,mod,"Lambda",L,"PRC",abs(diff), "MxMode", int(self.TXmode(L,thi,ca,X)[0]), "Angle - Crit",T-self.critang(L), "Stps",sp,"/",mxp)
                else:   # The desired precision is reached at least by 25% (of sig. digits)
                    k0=2*Pi/L               
                    N=self.TXmode(L,T,X)[1] #This is the real effective refractive index, kx/k0
                    kx=k0*N                 #The propagation constant (beta)
                    ki=k0*self.n1Ii(L)/max(self.smk,np.sin(T))        #This is the imaginary of the modes refractive index, kappa*k0.  (Big K= k0*kappa / sin(T))
                    #kyl=np.sqrt( kx**2 - k0**2)    #the decay constant in air, outside WG
                    #kyc=np.sqrt( - kx**2 + (kx/np.sin(T))**2)  #The decay constant in the WG core
                    #conf=self.getConfinement(kyc,kyl,self.Thickness,mod)
                    #ki2=k0*self.n1Ii(L)*conf
                    #arr.append([mod,L,kx,ki,k0,conf,ki2])            #Uncomment this and commented lines above to also calculate the confinement factor eta
                    arr.append([mod,L,kx,ki,k0])            #We also append them to the return array.           
        return np.asarray(arr)
    
##### FILE EXPORT / IMPORT FUNCTIONS #####    
    def makeIndex(self, array):
        Lsta,Lend,Ldel,j=10,0,100,-1             #Deduce the Modedata parameters from the beta array
        LL=0.0          #Last lambda
        idx=[]          #Mode index array
        for line in array:  #Read modedata line by line, line[1] is lambda
            j+=1
            lamb=round(line[1],5) #Round lambda to angstrom/10
            if (LL) > 0:
                if lamb-LL > 0: #Find largest delta Lambda
                    Ldel=min(Ldel,abs(lamb-LL))
                Lsta=min(Lsta,lamb)
                Lend=max(Lend,lamb)
                if LL >= lamb:
                    idx.append(j)
            else:
                Lsta=lamb
                idx.append(j)
            LL=lamb
        idx.append(len(array)-1)
        self.Lsta=Lsta
        self.Lend=Lend
        self.Ldel=Ldel
        self.mindex=idx


    def Export(self,name="output"):             #Write the beta array to a file (name).dat
        if len(self.beta) < 1:
            print("ERR: please generate data first with OBJ.generate()")
        else:
            file=open("{}.dat".format(name), 'w')
            file.write('Modenumber, Lambda (um), beta  (per um), alpha/2  (per um), k0  (per um); Thickness={} um\n'.format(self.Thickness))
            file.close()
            file=open("{}.dat".format(name), 'a')
            for line in self.beta:
                if len(line) == 7:
                    string="{:.0f},{:.4f},{:.12E},{:.12E},{:.12E},{:.12E},{:.12E}\n".format(*line)
                if len(line) == 5:
                    string="{:.0f},{:.4f},{:.12E},{:.12E},{:.12E}\n".format(*line)
                file.write(string)
            file.close()
            #np.savetxt("{}.dat".format(name),self.beta,delimiter=',')
    
    def Import(self,name):
        with open("{}.dat".format(name)) as file: #Read thickness of the WG, since this cannot be post-computed
            self.Thickness = next(file).split(";")[1].split("=")[1]
            chk=next(file)
        file.close()
        if chk.find(",") != -1:                  # Check if data is space or comma delimited, and import
            modes=np.loadtxt("{}.dat".format(name),delimiter=',',skiprows=1)
        else:
            modes=np.loadtxt("{}.dat".format(name), skiprows=1)
        Lsta,Lend,Ldel,j=10,0,100,-1             #Deduce the Modedata parameters from the beta array
        LL=0.0          #Last lambda
        idx=[]          #Mode index array
        for line in modes:  #Read modedata line by line, line[1] is lambda
            j+=1
            lamb=round(line[1],5) #Round lambda to angstrom/10
            if (LL) > 0:
                if lamb-LL != 0: #Find smallest delta Lambda
                    Ldel=min(Ldel,abs(lamb-LL))
                Lsta=min(Lsta,lamb)
                Lend=max(Lend,lamb)
                if LL >= lamb:
                    idx.append(j)
            else:
                Lsta=lamb
                idx.append(j)
            LL=lamb
        idx.append(len(modes)-1)
        self.Lsta=Lsta
        self.Lend=Lend
        self.Ldel=Ldel
        self.mindex=idx
        self.beta=modes
        
    def ImportLumerical(self,name="output.txt",thickness=1,columnspermode=7,firstbetacolumn=4,firstalphacolumn=5):         
    #Lumerical txt output files for modedata usually has lambda in first column and then a couple of columns per mode, in this case 7,             
    #where propagation constant beta is first found in col 4, and alhpa/2 in col. 5.
        try: 
            file=open("{}".format(name), 'r') #Read thickness of the WG, since this cannot be post-computed
            s = next(file)
            chk = next(file)
            file.close()
            if chk.find(",") != -1:                  # Check if data is space or comma delimited, and import
                modes=np.loadtxt("{}".format(name),delimiter=',',skiprows=1)
            else:
                modes=np.loadtxt("{}".format(name),skiprows=1)
        except IOError:
            print("Mode data not accessible: {}\n Not Found / Wrong delimiter? ".format(name))
        #print(modes)
        #Now Lets make a proper array: columns: MODE#, Lambda (um), beta (1/um), alpha/2 (1/um), k0 (1/um)
        #We extract two arrays, one only for TIR modes and one with all.
        c=cnt.c     #SPEED OF LIGHT
        Pi=cnt.pi   #circle number
        modenum=int((len(modes[0])-1)/columnspermode)
        tirmodes,trmodes=[],[]  #tirmodes holds total internal reflection modes, trmodes is a sub-array thereof
        allmodes=[]     # allmodes holds all modes
        n=0             #modenumber of the guided mode
        for m in range(0,modenum):  #For every mode
            freq=modes[:,0]     #Frequency array
            lamb=c/freq*10**6   #Wavelength array (microns)
            beta=modes[:,firstbetacolumn+m*columnspermode]/10**6
            alpha=modes[:,firstalphacolumn+m*columnspermode]/10**6
            k0=2*Pi/(lamb)
            i,j=0,0
            #print(lamb[1],beta[1],alpha[1])
            while i+1 < len(lamb):
                allmodes.append([int(m),lamb[i],beta[i],alpha[i],k0[i]])
                if beta[i] >= k0[i]:    #Check if mode point is in-between the dispersion lines of air/si
                    trmodes.append([int(n),lamb[i],beta[i],alpha[i],k0[i]])        #Why alpha/2? This is loss of the mode, so 2x bigger than k_i
                    j+=1
                i+=1
            if j > 2:   #Take only modes with at least 2 datapoints into account (2nm wide)
                n+=0.5
                if m==0 or len(tirmodes)<2:
                    tirmodes=np.asarray(trmodes)
                else:
                    tirmodes=np.vstack([tirmodes,np.asarray(trmodes)])  #Stack TIR modes into the output array
            trmodes=[]    
        #       print(m,'mode is TIR.,',j,'vs',len(beta))               # Check Datapoints per mode
        tirmodes=np.asarray(tirmodes)
        allmodes=np.asarray(allmodes)  
        self.makeIndex(tirmodes)          
        # Lsta,Lend,Ldel,j=10,0,100,-1             #Deduce the Modedata parameters from the beta array
        # LL=0.0          #Last lambda
        # idx=[]          #Mode index array
        # for line in tirmodes:  #Read modedata line by line, line[1] is lambda
            # j+=1
            # lamb=round(line[1],5) #Round lambda to angstrom/10
            # if (LL) > 0:
                # if lamb-LL != 0: #Find smallest delta Lambda
                    # Ldel=min(Ldel,abs(lamb-LL))
                # Lsta=min(Lsta,lamb)
                # Lend=max(Lend,lamb)
                # if LL >= lamb:
                    # idx.append(j)
            # else:
                # Lsta=lamb
                # idx.append(j)
            # LL=lamb
        # idx.append(len(modes)-1)
        # self.Thickness=thickness
        # self.Lsta=Lsta
        # self.Lend=Lend
        # self.Ldel=Ldel
        # self.mindex=idx
        self.beta=tirmodes
        self.all=allmodes
    
#### PLOT FUNCTIONS ####
    def plDisp(self,name,Mmin=0,Mmax=-1,kmin=0,kmax=-1,noplot=0,ret=0):  #Plotfunction of dispersion/absorption for beta vs. lambda for thickness h (um), name output name
        S=self
        Lambs=[S.Lsta,S.Lend,S.Ldel]
        x=np.asarray(np.arange(*Lambs))
        norm=mpl.colors.Normalize(vmin=0.0, vmax=1.0)
        fig, ax = plt.subplots(1,1)     
        Mmin=2*Mmin
        if Mmax==-1 or 2*Mmax+1>len(S.mindex):
            Mmax=int((len(S.mindex)-1)) 
        else:
            Mmax=int(2*Mmax)
        if kmax==-1:
            kmax=max(S.beta[:,2])
        elif kmin != 0:
            ax.plot(x,kmin*x/x,'--',color='darkgrey',label='$k_1$={}'.format(kmin))
            ax.plot(x,kmax*x/x,'-',color='darkgrey',label='$k_2$={}'.format(kmax))
        else:
            ax.plot(x,kmin*x/x,'--',color='darkgrey',label='$k_1$={}'.format(kmin))
            ax.plot(x,kmax*x/x,'-',color='darkgrey',label='$k_2$={}'.format(kmax))
        if Mmax-Mmin>=0:
            for m in range(Mmin,Mmax):
                xy=[]
                [ xy.append([S.beta[g,1],S.beta[g,2],S.beta[g,3]]) for g in range(S.mindex[m],S.mindex[m+1]) ]
                xz=np.asarray(xy)
                if kmax-kmin>=0 and len(xz)>1:
                    zz=[]
                    for i in range(len(xz)):
                        if xz[i,1]>=kmin and xz[i,1]<=kmax:
                            zz.append(xy[i])
                xyz=np.asarray(zz)
                if len(xyz)>1:
                    yi=scinter.interp1d(xyz[:,0],xyz[:,1]) #, fill_value='extrapolate')
                    zi=scinter.interp1d(xyz[:,0],xyz[:,2]) #, fill_value='extrapolate')
                    xi=np.arange((xyz[0,0]),(xyz[-1,0]),0.0001)
                    #ax.scatter(xyz[:,0],xyz[:,1], marker='o',s=1,edgecolors='none', c=(1-np.exp(-2*xyz[:,2])), cmap=plt.cm.viridis, norm=norm)
                    ax.scatter(xi,yi(xi), marker='o',s=1,edgecolors='none', c=(1-np.exp(-2*zi(xi))), cmap=plt.cm.viridis, norm=norm)
            
            ax.plot(x,2*cnt.pi*S.n1Ri(x)/x, '-', color='black', label='h={}, {}'.format(S.Thickness,S.Layers[1]))
            ax.plot(x,2*cnt.pi*S.n2Ri(x)/x, '-', color='brown', label=S.Layers[0])
            ax.plot(x,2*cnt.pi*S.n3Ri(x)/x, '--', color='blue', label=S.Layers[2])
            ax.set_xlabel('Wavelength / $\mu$m')
            ax.set_ylabel('K$_{//}$ / $\mu$m$^{-1}$ ')
            ax.set_xlim(Lambs[0],Lambs[1])
            ax.tick_params(
                axis='x', direction='in')
            ax.tick_params(
                axis='x', which='minor', direction='in')
            ax.tick_params(
                axis='y', direction='in')
            ax.tick_params(
                axis='y', which='minor', direction='in')
            ax.tick_params(top=True, right=True)
            ax.tick_params(which='minor', top=True, right=True)
            ax.xaxis.set_minor_locator(AutoMinorLocator()) #FixedLocator(np.arange(-30,30,2))
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            #ax.set_ylim(0,max(yp))
            fig.colorbar(ax.collections[0],ax=ax,label='Attenuation per $\mu$m')
            #ax.grid()
            ax.legend()
            fig.set_size_inches(6,4)
            fig.savefig("Dispersion-{}.png".format(name), bbox_inches='tight', dpi=300)    
            if noplot==1:
                plt.close(fig)
        else:
            print("Error: No Valid Mode Range Provided: {}-{}".format(Mmin/2,Mmax/2))
        if ret==1:
            return(xyz)
            
    def plModecount(self, name="noname",noplot=0):           #Plotfunction for  # of Modes against WL
        S=self
        Lambs=[S.Lsta,S.Lend,S.Ldel]
        xdat=(np.arange(*Lambs))
        ydat=[S.mxmoL(x) for x in xdat]
        fig, ax = plt.subplots(1,1)
        ax.plot(xdat,ydat, '-', color='r',  label='')
        ax.set_xlabel('Wavelength / nm')
        ax.set_ylabel('Mode Count')
        ax.grid()   
        fig.savefig("ModecountVsLambda-{}.pdf".format(name), bbox_inches='tight')
        if noplot==1:
            plt.close(fig)
        
    def plRef(self,layer=1,noplot=0):    #    
        S=self
        if layer < 1 or layer >3:   
            print("Error: Please choose layer index: 1,2, or 3")    
        else:
            if layer==1:
                real,imag=S.n1Ri,S.n1Ii
                label=S.Layers[1]
            if layer==2:
                real,imag=S.n2Ri,S.n2Ii
                label=S.Layers[0]
            if layer==3:
                real,imag=S.n3Ri,S.n3Ii
                label=S.Layers[2]
            Lambs=[S.Lsta,S.Lend,S.Ldel]
            x=(np.arange(*Lambs))
            fig, ax = plt.subplots(1,1)
            ax.plot(x,real(x), '--', color='b', label='{}, real'.format(label))
            ax.plot(x,imag(x), '--', color='g', label='{}, imaginary'.format(label))
            ax.set_xlabel('Wavelength / um')
            ax.set_ylabel('Refractive Index')
            ax.grid()   
            ax.legend()
            fig.savefig("RefractiveIndex-{}.pdf".format(label), bbox_inches='tight')
            if noplot==1:
                plt.close(fig)
