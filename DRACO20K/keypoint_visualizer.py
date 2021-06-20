import numpy as np
import matplotlib.pyplot as plt
import imageio


if __name__ == "__main__":

    # directory_path = "../generated_data/02958343/baa2f488ec67a0a7c66e38c3984e156f/"
    # directory_path = "../generated_data/02691156/baa972f48cde1dc290baeef8ba5b93e5/"
    directory_path = "../generated_data/03001627/8c629a89e570c8776a9cd58b3a6e8ee5/"
    
    image_number = 0
    kps = np.load(directory_path + "frame_%08d_KeyPoints.npy" % image_number)
    im = plt.imread(directory_path + "frame_%08d_Color_00.png" % image_number)
    depth = imageio.imread(directory_path + "frame_%08d_Depth_00.exr" % image_number)[:, :, 0]
    
    
    plt.imshow(im)

    print(kps.shape)
    for i in range(kps.shape[0]):
        
        # If visibility is 1 display point in red
        if kps[i, 2] == 1:
            plt.plot(kps[i, 0], kps[i, 1], 'ro')
        else:
            plt.plot(kps[i, 0], kps[i, 1], 'bo')

    # Semantic ids
    # {0: "R_F_WHEELCENTER",
    #  1: "R_B_WHEELCENTER",
    #  2: "L_B_WHEELCENTER",
    #  3: "L_F_WHEELCENTER",
    #  4: "R_F_BUMPER",
    #  5: "R_B_BUMPER",
    #  6: "L_B_BUMPER",
    #  7: "L_F_BUMPER",
    #  8: "R_F_ROOFTOP",
    #  9: "R_B_ROOFTOP",
    # 10: "L_B_ROOFTOP",
    # 11: "L_F_ROOFTOP",
    # 12: "L_F_BONNET",
    # 13: "R_F_BONNET",
    # 14: "R_B_BONNET",
    # 15: "L_B_BONNET",
    # 16: "R_F_LIGHT",
    # 17: "R_B_LIGHT",
    # 18: "L_B_LIGHT",
    # 19: "L_F_LIGHT",
    # 20: "R_MIRROR",
    # 21: "L_MIRROR"}

    # semantic_id = 21
    # plt.plot(kps[semantic_id, 0], kps[semantic_id, 1], 'rv')
    
    plt.show()

    #