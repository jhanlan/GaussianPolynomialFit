import numpy as np
import math
from numpy import linalg


# Central Window Stuff
def GaussWeightC(sigma):
    index_diff = np.arange(-2*sigma,2*sigma+1)
    index_diffnorm = index_diff/sigma
    index_diff2 = index_diffnorm*index_diffnorm
    wind = np.exp(-.5*index_diff2)
    

    wind = wind/np.amax(wind)
    
    return wind

def SquareWeightC(sigma):
    return np.array((4*sigma+1)*[1])

def WindowC(y,i0,sigma,npoints):
    return np.array([y[i%npoints] for i in range(i0-2*sigma,i0+2*sigma+1)])

def LocalRegionC(x,y,dy,weights,i0,sigma,periodic):
    #The idea is to carefully curate the data to just perform the fit that
    #is necessary. We will then stitch it back together as needed.
    
    npoints = len(y)
    if 0 in dy:
        print('one or more of the errors in the data are zero, setting the error to the average uncertainty')
        dy[np.argwhere(dy==0).flatten()] = np.average(dy)
    xrang = WindowC(x,i0,sigma,npoints)-x[i0]
    yrang = WindowC(y,i0,sigma,npoints)
    dyrang = WindowC(dy,i0,sigma,npoints)
    
    finweight = weights/(dyrang*dyrang)
    
    if not periodic:
        
        if i0<2*sigma:
            imin = 2*sigma-i0
            xrang = xrang[imin:]
            yrang = yrang[imin:]
            dyrang = dyrang[imin:]
            finweight = finweight[imin:]
        
        elif i0+2*sigma>npoints-1:
            imax = i0+2*sigma-npoints+1
            xrang = xrang[:-imax]
            yrang = yrang[:-imax]
            dyrang = dyrang[:-imax]
            finweight = finweight[:-imax]

    return xrang,yrang,dyrang,finweight


#Left Window Stuff
def GaussWeightL(sigma):
    index_diff = np.arange(-2*sigma,1)
    index_diffnorm = index_diff/sigma
    index_diff2 = index_diffnorm*index_diffnorm
    wind = np.exp(-.5*index_diff2)
    

    wind = wind/np.amax(wind)
    
    return wind
def SquareWeightLR(sigma):
    return np.array((2*sigma+1)*[1])

def WindowL(y,i0,sigma,npoints):
    return np.array([y[i%npoints] for i in range(i0-2*sigma,i0+1)])

def LocalRegionL(x,y,dy,weights,i0,sigma,periodic):
    
    npoints = len(y)
    if 0 in dy:
        print('one or more of the errors in the data are zero, setting the error to the average uncertainty')
        dy[np.argwhere(dy==0).flatten()] = np.average(dy)
    xrang = WindowL(x,i0,sigma,npoints)-x[i0]
    yrang = WindowL(y,i0,sigma,npoints)
    dyrang = WindowL(dy,i0,sigma,npoints)
    
    finweight = weights/(dyrang*dyrang)
    
    if not periodic:
        
        if i0<2*sigma:
            imin = 2*sigma-i0
            xrang = xrang[imin:]
            yrang = yrang[imin:]
            dyrang = dyrang[imin:]
            finweight = finweight[imin:]

    return xrang,yrang,dyrang,finweight



# Right Window Stuff
def GaussWeightR(sigma):
    index_diff = np.arange(2*sigma+1)
    index_diffnorm = index_diff/sigma
    index_diff2 = index_diffnorm*index_diffnorm
    wind = np.exp(-.5*index_diff2)
    
    wind = wind/np.amax(wind)
    
    return wind

def WindowR(y,i0,sigma,npoints):
    return np.array([y[i%npoints] for i in range(i0,i0+2*sigma+1)])

def LocalRegionR(x,y,dy,weights,i0,sigma,periodic):
    
    npoints = len(y)
    if 0 in dy:
        print('one or more of the errors in the data are zero, setting the error to the average uncertainty')
        dy[np.argwhere(dy==0).flatten()] = np.average(dy)
    xrang = WindowR(x,i0,sigma,npoints)-x[i0]
    yrang = WindowR(y,i0,sigma,npoints)
    dyrang = WindowR(dy,i0,sigma,npoints)
    
    finweight = weights/(dyrang*dyrang)
    
    if not periodic:
        
        if i0+2*sigma>npoints-1:
            imax = i0+2*sigma-npoints+1
            xrang = xrang[:-imax]
            yrang = yrang[:-imax]
            dyrang = dyrang[:-imax]
            finweight = finweight[:-imax]

    return xrang,yrang,dyrang,finweight


def RunSVD(x,y,dy,degree,weights):
    npoints = len(x)
    A = [[np.sqrt(weights[i])*x[i]**k/math.factorial(k) for k in range(degree)] for i in range(npoints)]
    b = np.sqrt(weights)*y
    
    u, s, vh = linalg.svd(A, full_matrices = False)
    vh = vh.transpose()
    
    a = np.zeros((1,degree))
    
    da = np.zeros((1,degree))

    for i in range(degree):
        
        
        uvector = np.zeros((1,npoints))
        vvector = np.zeros((1,degree))
        
        
        for k in range(npoints):
            uvector[0][k] = u[k][i]
        
        for k in range(degree):
            vvector[0][k] = vh[k][i]
        
        if s[i]!=0:
            a += np.matmul((np.matmul(uvector,b)/s[i]),vvector)
        
        s2 = 0
        
        for k in range(degree):
            
            s2 += (vh[i][k]/s[k])**2
        da[0][i] = np.sqrt(s2)
    
    return a, da



def PolyFit(x,y,dy=None,sigma=5,degree=3,periodic=False,window_shape='Gauss'):
    runagain= False
    if dy is None:
        print('No uncertainty added, inferring uncertainty from initial fit.')
        dy = len(y)*[1]
        runagain=True
    x = np.array(x)
    y = np.array(y)
    dy = np.array(dy)
    #Initialize arrays
    alistC = []
    dalistC = []

    alistL = []
    dalistL = []

    alistR = []
    dalistR = []
    
    npoints = len(x)
    
    ### Grab Weights ###
    if window_shape=='Gauss':
        weightsC = GaussWeightC(sigma)
        weightsL = GaussWeightL(sigma)
        weightsR = GaussWeightR(sigma)
    elif window_shape=='Square':
        weightsC = SquareWeightC(sigma)
        weightsL = SquareWeightLR(sigma)
        weightsR = SquareWeightLR(sigma)
    else:
        raise Exception('window_shape must be either "Gauss" or "Square"')
    
    ### Run Center window first
    for i0 in range(npoints):
        xC,yC,dyC,totalWeightC = LocalRegionC(x,y,dy,weightsC,i0,sigma,periodic)
        aC,daC = RunSVD(xC,yC,dyC,degree,totalWeightC)
        alistC.append(aC)
        dalistC.append(daC)
        
    if runagain:
        y1 = [alistC[i][0][0] for i in range(len(alistC))]
        dy = [np.sqrt(np.average((y-y1)**2))]*len(y1)
        
        alistC = []
        dalistC = []
        for i0 in range(npoints):
            xC,yC,dyC,totalWeightC = LocalRegionC(x,y,dy,weightsC,i0,sigma,periodic)
            aC,daC = RunSVD(xC,yC,dyC,degree,totalWeightC)
            alistC.append(aC)
            dalistC.append(daC)
    
    ### Run Left Window
    for i0 in range(degree-1,npoints):
        xL,yL,dyL,totalWeightL = LocalRegionL(x,y,dy,weightsL,i0,sigma,periodic)
        aL,daL = RunSVD(xL,yL,dyL,degree,totalWeightL)
        alistL.append(aL)
        dalistL.append(daL)
    
    ### Run Right Window
    for i0 in range(npoints-degree+1):
        xR,yR,dyR,totalWeightR = LocalRegionR(x,y,dy,weightsR,i0,sigma,periodic)
        aR,daR = RunSVD(xR,yR,dyR,degree,totalWeightR)
        alistR.append(aR)
        dalistR.append(daR)
    
    return np.array(alistC),np.array(dalistC),np.array(alistL),np.array(dalistL),np.array(alistR),np.array(dalistR)



def ReturnOriginalVariables(x,aC,daC,aL,daL,aR,daR,periodic=False):
    yCs = []
    yLs = []
    yRs = []
    dyCs = []
    dyLs = []
    dyRs = []
    degree = len(aC[0][0])
    for deg in range(degree):
        yCs.append([aC[i][0][deg] for i in range(len(aC))])
        yLs.append([aL[i][0][deg] for i in range(len(aL))])
        yRs.append([aR[i][0][deg] for i in range(len(aR))])
        
        dyCs.append([daC[i][0][deg] for i in range(len(daC))])
        dyLs.append([daL[i][0][deg] for i in range(len(daL))])
        dyRs.append([daR[i][0][deg] for i in range(len(daR))])
    xC = np.copy(x)
    if periodic:
        xL = np.copy(x)
        xR = np.copy(x)
    else:
        xL = xC[degree-1:]
        xR = xC[:-(degree-1)]
    return xC,xL,xR,np.array(yCs),np.array(yLs),np.array(yRs),np.array(dyCs),np.array(dyLs),np.array(dyRs)




