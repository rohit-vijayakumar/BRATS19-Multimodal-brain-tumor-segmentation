# Defining function for post-processing data
def paint_color_algo(pred_full, pred_core , pred_ET , li):   
        # first put the pred_full on T1c
        pred_full[pred_full > 0.2] = 2      # 240x240
        pred_full[pred_full != 2] = 0
        pred_core[pred_core > 0.2] = 1      # 64x64
        pred_core[pred_core != 1] = 0
        pred_ET[pred_ET > 0.2] = 4          # 64x64
        pred_ET[pred_ET != 4] = 0

        total = np.zeros((1,240,240),np.float32)  
        total[:,:,:] = pred_full[:,:,:]
        for i in range(pred_core.shape[0]):
            for j in range(64):
                for k in range(64):
                    if pred_core[i,0,j,k] != 0 and pred_full[0,li[i][0]+j,li[i][1]+k] !=0:
                        total[0,li[i][0]+j,li[i][1]+k] = pred_core[i,0,j,k]
                    if pred_ET[i,0,j,k] != 0 and pred_full[0,li[i][0]+j,li[i][1]+k] !=0:
                        total[0,li[i][0]+j,li[i][1]+k] = pred_ET[i,0,j,k]



        return total
