import open3d as o3d
from PIL import Image
import numpy as np
import torch
from unidepth.utils import colorize
from unidepth.models import UniDepthV1
from unidepth.utils.visualization import save_file_ply
from unidepth.utils.geometric import project_points


model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

rgb = np.array(Image.open('pair2/im0.png'))
rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
predictions = model.infer(rgb_torch)
depth1 = predictions["depth"].squeeze().cpu().numpy()

depth_pred_col = colorize(depth1, vmin=0.01, vmax=10.0, cmap="magma_r")
Image.fromarray(depth_pred_col).save("original.png")

H, W = rgb.shape[:2]
torch_intrinsic1 = predictions["intrinsics"] # Shape: (B, 3, H,W)
point_cloud_flat =  predictions["points"].view(1, 3, -1)  # Shape: (B, 3, HW)
point_cloud_transposed = point_cloud_flat.transpose(1, 2)  # Shape: (B, HW, 3)

points = predictions["points"].squeeze().cpu().permute(1,2,0).numpy().reshape(H*W, 3)
new_rgb = rgb.reshape(H*W, 3)
file = "im1.ply"
save_file_ply(points, new_rgb, file)

pcd1 = o3d.io.read_point_cloud("im1.ply")
points = np.asarray(pcd1.points)
points_tensor = torch.tensor(points, dtype=torch.float)
points_tensor = points_tensor.view(-1, 3).unsqueeze(0).cuda()

depth = project_points(point_cloud_transposed, torch_intrinsic1, (H,W))
depth_pred_col = colorize(depth.squeeze().cpu().numpy(), vmin=0.01, vmax=10.0, cmap="magma_r")
Image.fromarray(depth_pred_col).save("averaged1.png")

depth = project_points(points_tensor, torch_intrinsic1, (H,W))
depth_pred_col = colorize(depth.squeeze().cpu().numpy(), vmin=0.01, vmax=10.0, cmap="magma_r")
Image.fromarray(depth_pred_col).save("averaged2.png")
