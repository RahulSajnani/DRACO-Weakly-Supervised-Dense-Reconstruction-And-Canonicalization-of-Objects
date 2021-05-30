import numpy as np
import matplotlib.pyplot as plt
import cv2
from visualize_nocs import *
import glob
if __name__=="__main__":


    files = sorted(glob.glob("/home/aadilmehdi/RRC/WSNOCS/14102020/cars/ValSet/val/02958343/1c7a2752e1d170e099399ee63318a21b/frame_00000000_C3DPO_00.png"))
    for f in files:
        nocs_image = plt.imread(f)[:, :, :3]
        print(nocs_image)

        #plane_rotation_mat = np.array([[ 0.12421331, -1.06393214,  0.23896093],
    #                        [-0.93976668,  0.06419056,  0.50506007],
    #                        [ -0.49749532, -0.23067563, -0.89684068]])


        car_rotation_mat = np.array(
            [[ 0.12266913, -0.93841971, -0.14111399],#, -0.01749591],
            [-0.94150793, -0.10757955, -0.09544285],#, -0.00433662],
            [ 0.07557283,  0.15391763, -0.961159  ],#,  0.0075859 ],
            #[ 0.        ,  0.        ,  0.        ,  1.        ]]
            ]
            )



        # rotation_mat = np.array([[ 0.10401288, -0.97621297, -0.13562396],
        #                         [-0.98353821, -0.09394213, -0.07730554],
        #                         [ 0.06216511,  0.14517352, -0.98294882]])
        
        nocs_image = rotate_nocs(nocs_image, car_rotation_mat)
        # plt.imsave("nocs_image_12.jpg", nocs_image)
        cv2.imwrite("nocs_image_12.jpg",nocs_image*255)# cv2.cvtColor( (nocs_image*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        plt.imshow(nocs_image)
        plt.show()


    # gt_files = sorted(glob.glob("../../../../Val/*"))
    # pipeline1_files = sorted(glob.glob("../../../../Val/*"))
    # pipeline2_files = sorted(glob.glob("../../../../Val/*"))
    # image = None

    # C1 = []
    # C2 = []

    # R = np.asarray(
    #     [
    #         [ 1, 0, 0],
    #         [ 0, 1, 0],
    #         [ 0, 0, 1],
    #     ]
    # )

    # c1 = []
    # c2 = []
    # for i in range(len(gt_files)):
    #     gt = sorted(glob.glob( gt_files[i] + '/*_nox00_gt.png'))
    #     p1 = sorted(glob.glob(pipeline1_files[i] + '/*_nox00_pred.png'))
    #     p2 = sorted(glob.glob(pipeline2_files[i] + '/*nox00_pred.png'))

    #     for j in range(len(gt)):
    #         nocs_image_gt = gt[j]
    #         nocs_image_p1 = p1[j]
    #         nocs_image_p2 = p2[j]

    #         nocs_gt = read_image(nocs_image_gt)
    #         nocs_p1 = read_image(nocs_image_p1)
    #         nocs_p2 = read_image(nocs_image_p2)

    #         # nocs_gt = rotate_nocs(nocs_gt,R)

    #         print(np.abs(np.mean(nocs_gt-nocs_p1)))
    #         print(np.abs(np.mean(nocs_gt-nocs_p2)))
    #         print()
    #         c1.append(np.abs(np.mean(nocs_gt-nocs_p1)))
    #         c2.append(np.abs(np.mean(nocs_gt-nocs_p2)))

    # c1 = np.array(c1)
    # c2 = np.array(c2)
    # print("a", np.mean(c1))
    # # for b in c2:
    # #     print(b)
    # print("b", np.nanmean(c2))

    # #         pcd_gt = visualize_nocs_map(nocs_gt, image)
    # #         pcd_p1 = visualize_nocs_map(nocs_p1, image)
    # #         pcd_p2 = visualize_nocs_map(nocs_p2, image)

    # #         T = R @ (np.asarray(pcd_gt.points) - 0.5).T + 0.5 
    # #         pcd_gt.points = o3d.utility.Vector3dVector(T.T)

    # #         pcd_gt_points = np.asarray(pcd_gt.points)
    # #         pcd_p1_points = np.asarray(pcd_p1.points)
    # #         pcd_p2_points = np.asarray(pcd_p2.points)

    # #         pcd_gt_points = torch.from_numpy(pcd_gt_points.astype(float)).unsqueeze(0)
    # #         pcd_p1_points = torch.from_numpy(pcd_p1_points.astype(float)).unsqueeze(0)
    # #         pcd_p2_points = torch.from_numpy(pcd_p2_points.astype(float)).unsqueeze(0)

    # #         c1, _ = chamfer_distance(pcd_gt_points.float(), pcd_p1_points.float())
    # #         c2, _ = chamfer_distance(pcd_gt_points.float(), pcd_p2_points.float())

    # #         C1.append(c1)
    # #         C2.append(c2)

    # #         print(nocs_image_gt)
    # #         print(nocs_image_p1)
    # #         print(c1)
    # #         print(nocs_image_p2)
    # #         print(c2)
    # #         print()
    # # C1 = np.array(C1)
    # # C2 = np.array(C2)
    # # print("Average Pipeline 1: ", C1.mean())
    # # print("Average Pipeline 2: ", C2.mean())
