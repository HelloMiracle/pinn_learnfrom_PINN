import os
import imageio


def create_gif(image_list, gif_name):
    frames = []
    for image_name in image_list:
        if image_name.endswith('.jpg'):
            print(image_name)
            frames.append(imageio.imread(image_name))
    # Save them as frames into a gif
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.1)

    return
def main():

    path = r'./check_P/'  # 存放PNG图片文件夹位置
    files = os.listdir(path)
    # files.sort()
    files.sort(key=lambda x: int(x[:-4]))
    print(files)
    image_list = [path + img for img in files]

    gif_name = 't_0_100_p.gif'  # 生成gif的名称
    create_gif(image_list, gif_name)

if __name__ == "__main__":
    main()