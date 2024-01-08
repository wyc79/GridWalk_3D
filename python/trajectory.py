import os

import deeplabcut as dlc

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

root = '../GridWalking-wyc-2023-10-30'

video_path = os.path.join(root, 'videos')
out_path = os.path.join(root, 'dlc_analyzed')

#create a path variable that links to the config file:
config_path = f"{root}/config.yaml"

print(os.path.exists(config_path))

videofile_path = f'{root}/videos'

dlc.filterpredictions(config_path, videofile_path, destfolder=out_path, )
# https://github.com/DeepLabCut/DeepLabCut/blob/main/docs/standardDeepLabCut_UserGuide.md
dlc.plot_trajectories(config_path, videofile_path, destfolder=out_path, filtered=True,
                      displayedbodyparts=['mid_spine'])

