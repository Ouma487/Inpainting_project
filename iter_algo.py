from Patch_priorities import *
from tools import *

def iterate(image , firstmask , patch_size):
    mask=np.copy(firstmask)
    frt = init_bord_m(mask)
    new_image= delete_zone(image,mask)
    #view_data(new_image)
    K=0
    ps=patch_size//2
    
    while (len(frt)>0):
        p_point= maxP(new_image,mask,frt,patch_size)
        p_patch=get_patch(new_image,p_point,patch_size)
        new_patch = get_patch(new_image, (ps,ps),patch_size)
        d = similarity(new_patch,p_patch,get_patch(mask,(ps,ps),patch_size))
        #print(d)
        chosenX,chosenY = p_point
        
        maskpatch = get_patch(mask,(chosenX,chosenY),patch_size)
        # Looking for a patch that is the closest to the content of uncomplete patch
        for x in range(ps, new_image.shape[0]-ps):
            for y in range(ps,new_image.shape[1]-ps):
                potential = True
                firstpatch=get_patch(firstmask,(x,y),patch_size)
                for i in range(patch_size):
                    for j in range(patch_size):
                        if(firstpatch[i,j]==0):
                            potential=False
                if (potential):
                    testPatch = get_patch(new_image,(x,y),patch_size)
                    dtest = similarity(p_patch,testPatch,maskpatch)
                    if dtest < d :
                        d = dtest
                        #print(dtest)
                        #print('coordinates are {0} {1}'.format(x,y))
                        new_patch = np.copy(testPatch)
        #view_data(new_patch)
        # Filling
        for i in range(-ps,ps+1) :
            for j in range(-ps,ps+1):
                if(mask[chosenX+i,chosenY+j]==0):
                    new_image[chosenX+i,chosenY+j]= new_patch[ps+i,ps+j]
                    mask[chosenX+i,chosenY+j]= 1
        frt=init_bord_m(mask)
        
        cv2.imwrite("output/"+str(K)+".jpg", new_image)
        if K % 10 ==0:
            
            print(f"Iteration {K} completed")
            view_data(new_image)
        
        K+=1
    
    view_data(new_image)