import os
import cv2


def mp4_generator(img_path, save_path):
    """mp4生成函数"""
    files = os.listdir(img_path)
    files.sort()
    print(files)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    image = cv2.imread(img_path + files[0])
    height, width, layer = image.shape
    video_writer = cv2.VideoWriter(save_path, fourcc, 0.5, (width, height))
    for file in files:
        img = cv2.imread(img_path + file)
        video_writer.write(img)
    print("图片转视频结束！")
    video_writer.release()
    cv2.destroyAllWindows()


path1 = 'G:/jj_st/one_mode_graph/st/jjzxx/'
path2 = 'G:/jj_st/one_mode_graph/st/closeness_centrality.mp4'
mp4_generator(path1, path2)
