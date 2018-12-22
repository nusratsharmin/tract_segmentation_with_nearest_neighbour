"""Nearest Neighbour applied for bundle segmentation  
"""
import numpy as np
import time
from dipy.tracking.streamline import set_number_of_points
from pykdtree.kdtree import KDTree
from dipy.tracking.vox2track import streamline_mapping
import nibabel as nib

def dsc(estimated_tract, true_tract):
    """Compute the overlap between the segmented tract and ground truth tract
    """
    aff=np.array([[-1.25, 0, 0, 90],[0, 1.25, 0, -126],[0, 0, 1.25, -72],[0, 0, 0, 1]])
    voxel_list_estimated_tract = streamline_mapping(estimated_tract, affine=aff).keys()
    voxel_list_true_tract = streamline_mapping(true_tract, affine=aff).keys()
    TP = len(set(voxel_list_estimated_tract).intersection(set(voxel_list_true_tract)))
    vol_A = len(set(voxel_list_estimated_tract))
    vol_B = len(set(voxel_list_true_tract))
    DSC = 2.0 * float(TP) / float(vol_A + vol_B)
    return DSC    
         
def load(filename):
    """Load tractogram from TRK file 
    """
    wholeTract= nib.streamlines.load(filename)  
    wholeTract = wholeTract.streamlines
    return  wholeTract    

def resample(streamlines, no_of_points):
    """Resample streamlines using 12 points and also flatten the streamlines
    """
    return np.array([set_number_of_points(s, no_of_points).ravel() for s in streamlines]) 
    
def build_kdtree(points, leafsize):
    """Build kdtree with resample streamlines 
    """
    return KDTree(points,leafsize =leafsize)    
    
def kdtree_query(tract,kd_tree):
    """compute 1 NN using kdtree query and return the id of NN
    """
         
    dist_kdtree, ind_kdtree = kd_tree.query(tract, k=1)
    return np.hstack(ind_kdtree) 

def compute_dsc(estimated_tract, filename_true_tract):
    """Comparison between the segmented tract and ground truth tract
    """    
    true_tract=load(filename_true_tract) 
    return dsc(estimated_tract, true_tract)

def segmentation_with_NN(filename_tractogram, filename_example_tract,no_of_points,leafsize):

    """Nearest Neighbour applied for bundle segmentation 

    """   
    #load tractogram
    print("Loading tractogram: %s" %filename_tractogram)
    tractogram=load(filename_tractogram) 
        
    #load tract
    print("Loading example tract: %s" %filename_example_tract)
    tract=load(filename_example_tract) 
    
    t0=time.time()
    #resample whole tractogram
    print("Resampling tractogram........" )
    resample_tractogram=resample(tractogram,no_of_points=no_of_points)
        
    #resample example tract
    print("Resampling example tract.......")
    resample_tract=resample(tract,no_of_points=no_of_points)
    
    #build kdtree
    print("Buildng kdtree")
    kd_tree=build_kdtree (resample_tractogram, leafsize=leafsize)
    
    #kdtree query to retrive the NN id
    query_idx=kdtree_query(resample_tract, kd_tree)
    
    #extract the streamline from tractogram
    estimated_tract=tractogram[query_idx]  

    print("Total amount of time to segment the bundle is %f seconds" % (time.time()-t0))   
    return  estimated_tract
     
if __name__ == '__main__':
    
    print(__doc__)
    
    filename_tractogram="/data/HCP3-IU/derivatives/ensemble_tracking_EPR/sub-500222/sub-500222_var-EPR_tract.tck"
    
    filename_example_tract="/data/HCP3-IU/derivatives/afq_EPR/sub-512835/sub-512835_var-EPR_Left_Arcuate.trk" 
    
    filename_true_tract="/data/HCP3-IU/derivatives/afq_EPR/sub-500222/sub-500222_var-EPR_Left_Arcuate.trk" 

    # Main parameters:
    no_of_points=12 # number of points for resampling
    leafsize=10    #number of leaf size for kdtree
    
    print("Segmenting tract with NN......")   
    estimated_tract= segmentation_with_NN(filename_tractogram, 
                         filename_example_tract,
                         no_of_points,
                         leafsize)
                                             
      
    print("Computing Dice Similarity Coefficient......")               
    print ("DSC= %f" %compute_dsc(estimated_tract,
                        filename_true_tract))                      