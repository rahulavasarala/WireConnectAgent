import redis
import numpy as np
import open3d as o3d

redis_client = redis.Redis()

def main():

    blob = redis_client.get("point_cloud")
    points = np.frombuffer(blob, dtype=np.float32)

    assert points.size % 3 == 0, "Point cloud blob size not divisible by 3"
    points = points.reshape((-1, 3)).astype(np.float64)

    print("Points has been extracted!")
    print(points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    o3d.visualization.draw_geometries([pcd], window_name="Redis Point Cloud")
    pass

if __name__ == "__main__":
    main()