import numpy as np
from tqdm import tqdm

class Sampler:

    def __init__(self, H, W, patch_size, pad): #Channels x H x W
        
        
        img = np.zeros((1, H, W))
        dim_tot = patch_size+2*pad

        it0=np.arange(0, H, patch_size)
        it1=np.arange(0, W, patch_size)

        patch_start_x = []
        patch_start_y = []
        patch_end_x = []
        patch_end_y = []

        mask_start_x = []
        mask_start_y = []
        mask_end_x = []
        mask_end_y = []


        outputPatch_start_x = []
        outputPatch_start_y = []
        outputPatch_end_x = []
        outputPatch_end_y = []


        for j in tqdm(it1):
            for i in it0:
                
                index1 = H-(i+patch_size)
                index2 = W-(j+patch_size)

                # ---------------------------- FIST COLUMN --------------------------
                
                if (j==it1[0]):
                    

                    patch_start_y.append(j)
                    mask_start_y.append(j)
                    mask_end_y.append(j+patch_size)
                    outputPatch_end_y.append(patch_size)
                    outputPatch_start_y.append(0)

                    # FIRST ROW
                    if (i==it0[0]):
                        patch_start_x.append(i)
                        patch_end_x.append(i+dim_tot)
                        patch_end_y.append(j+dim_tot)
                        mask_start_x.append(i)
                        mask_end_x.append(i+patch_size)
                        outputPatch_start_x.append(0)
                        outputPatch_end_x.append(patch_size)
                    
                    # LAST ROW
                    elif (i==it0[-1]):
                        patch_start_x.append(H -(dim_tot))
                        patch_end_x.append(H)
                        patch_end_y.append(j+dim_tot)
                        outputPatch_end_x.append(dim_tot)

                        if (H%patch_size)==0:
                            mask_start_x.append(dim_tot-patch_size)
                            mask_end_x.append(dim_tot)
                            outputPatch_start_x.append(dim_tot-patch_size)
                        else:
                            mask_start_x.append(H-(H%patch_size))
                            mask_end_x.append(H)
                            outputPatch_start_x.append(dim_tot-(H%patch_size))
                    
                    # CENTRAL ROW
                    else:
                        
                        patch = img[:,i-pad:i+patch_size+pad,j:j+dim_tot] 
                        patch_end_y.append(j+dim_tot)
                        mask_end_x.append(i+patch_size)
                        mask_start_x.append(i)


                        if patch.shape[1]< (dim_tot):
                            patch_start_x.append(H-(dim_tot))
                            patch_end_x.append(H)
                            outputPatch_start_x.append(dim_tot-index1-patch_size )
                            outputPatch_end_x.append(dim_tot-index1)
                        else:
                            patch_start_x.append(i-pad)
                            patch_end_x.append(i+patch_size+pad)
                            outputPatch_start_x.append(pad)
                            outputPatch_end_x.append(dim_tot-pad)


                # --------------------- LAST COLUMN -----------------------

                elif (j==it1[-1]):

                    patch_end_y.append(W)
                    mask_end_y.append(W)
                    outputPatch_end_y.append(dim_tot)
                    patch_start_y.append(W-(dim_tot))

                    # -----FIRST ROW
                    if (i==it0[0]):

                        patch_start_x.append(i)
                        patch_end_x.append(i+dim_tot)
                        outputPatch_start_x.append(0)
                        outputPatch_end_x.append(patch_size)
                        mask_start_x.append(i)
                        mask_end_x.append(i+patch_size)

                        if (W%patch_size)==0:
                            mask_start_y.append(W-patch_size)
                            outputPatch_start_y.append(dim_tot-patch_size)
                        else:
                            mask_start_y.append(W-(W%patch_size))
                            outputPatch_start_y.append(dim_tot-(W%patch_size))


                    # -----LAST ROW
                    elif (i==it0[-1]):

                        patch_start_x.append(H-(dim_tot))
                        patch_end_x.append(H)
                        mask_end_x.append(H)
                        outputPatch_end_x.append(dim_tot)


                        if (W%patch_size)==0:
                            mask_start_y.append(W-patch_size)
                            outputPatch_start_y.append(dim_tot-patch_size)
                            if (H%patch_size)==0:
                                mask_start_x.append(H-patch_size)
                                outputPatch_start_x.append(dim_tot-patch_size)
                            else:
                                mask_start_x.append(H-(H%patch_size))
                                outputPatch_start_x.append(dim_tot-(H%patch_size))
                        else:
                            mask_start_y.append(W-(W%patch_size))
                            outputPatch_start_y.append(dim_tot-(W%patch_size))
                            if (H%patch_size)==0:
                                mask_start_x.append(H-(patch_size))
                                outputPatch_start_x.append(dim_tot-patch_size)
                            else:
                                mask_start_x.append(H-(H%patch_size))
                                outputPatch_start_x.append(dim_tot-(H%patch_size))

                    # ----CENTRAL ROWS
                    else:
                        patch = img[:,i-pad:i+patch_size+pad,-(dim_tot):] 

                        mask_start_x.append(i)


                        mask_end_x.append(i+patch_size)

                        if patch.shape[1]< (dim_tot):

                            patch_start_x.append(H-dim_tot)
                            patch_end_x.append(H)
                            
                            outputPatch_start_x.append(dim_tot-index1-patch_size)
                            outputPatch_end_x.append(dim_tot-index1)
                            
                            if (W%patch_size)==0:
                                mask_start_y.append(W-patch_size)
                                outputPatch_start_y.append(dim_tot-patch_size)
                            else:
                                mask_start_y.append(W-(W%patch_size))
                                outputPatch_start_y.append(dim_tot-(W%patch_size))

                        else:
                            #outputPatch, outputRec = self.inference_filtering(patch)

                            patch_start_x.append(i-pad)
                            patch_end_x.append(i+patch_size+pad)
                            outputPatch_start_x.append(pad)
                            outputPatch_end_x.append(dim_tot-pad)
                            

                            if (W%patch_size)==0:
                                mask_start_y.append(W-patch_size)
                                outputPatch_start_y.append(dim_tot-patch_size)
                            else:
                                mask_start_y.append(W-(W%patch_size))
                                outputPatch_start_y.append(dim_tot-(W%patch_size))
                

                    # -------------------------- CENTRAL COLUMNS --------------------
                else:

                    mask_start_y.append(j)
                    # -----FIRST ROW
                    if (i==it0[0]):

                        patch = img[:,i:i+dim_tot,j-pad:j+patch_size+pad] 

                        patch_start_x.append(i)
                        patch_end_x.append(i+dim_tot)
                        mask_end_x.append(i+patch_size)
                        mask_end_y.append(j+patch_size)
                        mask_start_x.append(i)
                        outputPatch_start_x.append(0)
                        outputPatch_end_x.append(patch_size)

                        if patch.shape[2]< (dim_tot):

                            patch_start_y.append(W-(dim_tot))
                            patch_end_y.append(W)
                            
                            outputPatch_start_y.append(dim_tot-index2-patch_size)
                            outputPatch_end_y.append(dim_tot-index2)

                        else:
                            patch_start_y.append(j-pad)
                            patch_end_y.append(j+patch_size+pad)
                            outputPatch_start_y.append(pad)
                            outputPatch_end_y.append(dim_tot-pad)


                    # -----LAST ROW
                    elif (i==it0[-1]):

                        patch = img[:,-(dim_tot):,j-pad:j+patch_size+pad] 
                        patch_start_x.append(H-(dim_tot))
                        patch_end_x.append(H)
                        mask_end_y.append(j+patch_size)
                        outputPatch_end_x.append(dim_tot)

                        if patch.shape[2]< (dim_tot):

                            patch_start_y.append(W-(dim_tot))
                            patch_end_y.append(W)  
                            mask_start_x.append(i)
                            mask_end_x.append(i+patch_size)
                            
                            
                            outputPatch_start_y.append(dim_tot-index2-patch_size)
                            outputPatch_end_y.append(dim_tot-index2)

                            if (H%patch_size)==0:
                                outputPatch_start_x.append(dim_tot-patch_size)
                            else:
                                outputPatch_start_x.append(dim_tot-(H%patch_size))
                        else:

                            #outputPatch, outputRec = self.inference_filtering(patch)
                            patch_start_y.append(j-pad)
                            patch_end_y.append(j+patch_size+pad)

                            mask_end_x.append(H)

                            outputPatch_start_y.append(pad)
                            outputPatch_end_y.append(dim_tot-pad)

                            if (H%patch_size)==0:
                                mask_start_x.append(H-patch_size)
                                outputPatch_start_x.append(dim_tot-patch_size)
                            else:
                                mask_start_x.append(H-(H%patch_size))
                                outputPatch_start_x.append(dim_tot-(H%patch_size))


                    # -----CENTRAL ROW
                    else:
                        
                        patch = img[:,i-pad:i+patch_size+pad,j-pad:j+patch_size+pad] 

                        mask_start_x.append(i)
                        mask_end_x.append(i+patch_size)
                        mask_end_y.append(j+patch_size)


                        if patch.shape[1]< (dim_tot):

                            patch_start_x.append(H-(dim_tot))
                            patch_end_x.append(H)
                            outputPatch_start_x.append(dim_tot-index1-patch_size)
                            outputPatch_end_x.append(dim_tot-index1)

                            if patch.shape[2]< (dim_tot):
                                patch_start_y.append(W-(dim_tot))
                                patch_end_y.append(W)
                                outputPatch_start_y.append(dim_tot-index2-patch_size)
                                outputPatch_end_y.append(dim_tot-index2)
                            else:
                                patch_start_y.append(j-pad)
                                patch_end_y.append(j+patch_size+pad)
                                outputPatch_start_y.append(pad)
                                outputPatch_end_y.append(dim_tot-pad)

                        else:

                            patch_start_x.append(i-pad)
                            patch_end_x.append(i+patch_size+pad)
                            outputPatch_start_x.append(pad)
                            outputPatch_end_x.append(dim_tot-pad)

                            if patch.shape[2]< (dim_tot):
                                patch_start_y.append(W-dim_tot)
                                patch_end_y.append(W)
                                outputPatch_start_y.append(dim_tot-index2-patch_size)
                                outputPatch_end_y.append(dim_tot-index2)
                            else:
                                patch_start_y.append(j-pad)
                                patch_end_y.append(j+patch_size+pad)
                                outputPatch_start_y.append(pad)
                                outputPatch_end_y.append(dim_tot-pad)

        input = {
            'x': (patch_start_x, patch_end_x),
            'y': (patch_start_y, patch_end_y)
        }
        mask = {
            'x': (mask_start_x, mask_end_x),
            'y': (mask_start_y, mask_end_y)
        }
        output = {
            'x': (outputPatch_start_x, outputPatch_end_x),
            'y': (outputPatch_start_y,outputPatch_end_y)
        }

        self.sampler_input = input
        self.sampler_mask = mask
        self.sampler_output = output
        self.patch_size = patch_size
        self.pad = pad

    def apply(self, image, batch_size, transform, out_channels): #Channels x H x W

        dim_tot = self.patch_size+2*self.pad

        C = image.shape[0]
        H = image.shape[1]
        W = image.shape[2]

        patch_start_x, patch_end_x  = self.sampler_input['x']
        patch_start_y, patch_end_y  = self.sampler_input['y']
        mask_start_x, mask_end_x  = self.sampler_mask['x']
        mask_start_y, mask_end_y  = self.sampler_mask['y']
        outputPatch_start_x, outputPatch_end_x  = self.sampler_output['x']
        outputPatch_start_y, outputPatch_end_y  = self.sampler_output['y']

        batch_number=np.arange(0,len(patch_start_x),batch_size,dtype=(np.int64))

        res = np.zeros((out_channels,H,W), dtype = np.float32)
        for hh in batch_number:
            batch_stack_size = batch_size
            
            if (len(patch_start_x)%batch_size!=0) and (hh==batch_number[-1]): 
                batch_stack_size = len(patch_start_x)-hh
            
            batch_stack = np.zeros((batch_stack_size,C,dim_tot,dim_tot),dtype = np.float32)
            
            for qq in range(batch_stack.shape[0]):
                batch_stack[qq,:,:,:] = image[:,patch_start_x[hh+qq]:patch_end_x[hh+qq],patch_start_y[hh+qq]:patch_end_y[hh+qq]] 

            outputPatch = transform(batch_stack) #(5, 2, 256, 256)

            
            #res[:, mask_start_x[hh+qq]:mask_end_x[hh+qq],mask_start_y[hh+qq]:mask_end_y[hh+qq]] = outputPatch[qq, :, outputPatch_start_x[hh+qq]:outputPatch_end_x[hh+qq], outputPatch_start_y[hh+qq]:outputPatch_end_y[hh+qq]]
            
            for qq in range(batch_stack.shape[0]):
                res[:, mask_start_x[hh+qq]:mask_end_x[hh+qq],mask_start_y[hh+qq]:mask_end_y[hh+qq]] = outputPatch[qq, :, outputPatch_start_x[hh+qq]:outputPatch_end_x[hh+qq], outputPatch_start_y[hh+qq]:outputPatch_end_y[hh+qq]]

        return(res)


if __name__ == '__main__':

    patch_size = 128
    pad = patch_size // 2

    sampler = Sampler(512, 512, patch_size = patch_size, pad = pad)

    image = np.ones((3, 512, 512))
    batch_size = 4

    def transform(x):
        print(x.shape)
        exit(0)

    out_channels = 1

    out = sampler.apply(
        image,
        batch_size,
        transform,
        out_channels
    )
    print(out.shape)
