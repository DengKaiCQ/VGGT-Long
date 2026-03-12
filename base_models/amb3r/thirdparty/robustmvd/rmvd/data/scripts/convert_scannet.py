#!/usr/bin/env python2

# Script based on https://github.com/ScanNet/ScanNet/tree/master/SensReader/python

# References:
#   https://github.com/HengyiWang/amb3r/blob/main/thirdparty/robustmvd/rmvd/data/scripts/convert_scannet.py

import argparse
import os, struct, sys

import numpy as np
import zlib
import imageio
import cv2
from tqdm import tqdm

COMPRESSION_TYPE_COLOR = {-1:'unknown', 0:'raw', 1:'png', 2:'jpeg'}
COMPRESSION_TYPE_DEPTH = {-1:'unknown', 0:'raw_ushort', 1:'zlib_ushort', 2:'occi_ushort'}

class RGBDFrame():

  def load(self, file_handle):
    self.camera_to_world = np.asarray(struct.unpack('f'*16, file_handle.read(16*4)), dtype=np.float32).reshape(4, 4)
    self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
    self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
    self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
    self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
    self.color_data = ''.join(struct.unpack('c'*self.color_size_bytes, file_handle.read(self.color_size_bytes)))
    self.depth_data = ''.join(struct.unpack('c'*self.depth_size_bytes, file_handle.read(self.depth_size_bytes)))

  def decompress_depth(self, compression_type):
    if compression_type == 'zlib_ushort':
       return self.decompress_depth_zlib()
    else:
       raise

  def decompress_depth_zlib(self):
    return zlib.decompress(self.depth_data)

  def decompress_color(self, compression_type):
    if compression_type == 'jpeg':
       return self.decompress_color_jpeg()
    else:
       raise

  def decompress_color_jpeg(self):
    return imageio.imread(self.color_data)


class SensorData:

  def __init__(self, filename):
    self.version = 4
    self.load(filename)

  def load(self, filename):
    with open(filename, 'rb') as f:
      version = struct.unpack('I', f.read(4))[0]
      assert self.version == version
      strlen = struct.unpack('Q', f.read(8))[0]
      self.sensor_name = ''.join(struct.unpack('c'*strlen, f.read(strlen)))
      self.intrinsic_color = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
      self.extrinsic_color = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
      self.intrinsic_depth = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
      self.extrinsic_depth = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
      self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack('i', f.read(4))[0]]
      self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack('i', f.read(4))[0]]
      self.color_width = struct.unpack('I', f.read(4))[0]
      self.color_height =  struct.unpack('I', f.read(4))[0]
      self.depth_width = struct.unpack('I', f.read(4))[0]
      self.depth_height =  struct.unpack('I', f.read(4))[0]
      self.depth_shift =  struct.unpack('f', f.read(4))[0]
      num_frames =  struct.unpack('Q', f.read(8))[0]
      self.frames = []
      for i in range(num_frames):
        frame = RGBDFrame()
        frame.load(f)
        self.frames.append(frame)

  def export_depth_images(self, output_path, image_size=None, frame_skip=1):
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    print 'exporting', len(self.frames)//frame_skip, ' depth frames to', output_path
    for f in range(0, len(self.frames), frame_skip):
      depth_data = self.frames[f].decompress_depth(self.depth_compression_type)
      depth = np.fromstring(depth_data, dtype=np.uint16).reshape(self.depth_height, self.depth_width)
      if image_size is not None:
        depth = cv2.resize(depth, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
      imageio.imwrite(os.path.join(output_path, str(f) + '.png'), depth)

  def export_color_images(self, output_path, image_size=None, frame_skip=1):
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    print 'exporting', len(self.frames)//frame_skip, 'color frames to', output_path
    for f in range(0, len(self.frames), frame_skip):
      color = self.frames[f].decompress_color(self.color_compression_type)
      if image_size is not None:
        color = cv2.resize(color, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
      imageio.imwrite(os.path.join(output_path, str(f) + '.jpg'), color)

  def save_mat_to_file(self, matrix, filename):
    with open(filename, 'w') as f:
      for line in matrix:
        np.savetxt(f, line[np.newaxis], fmt='%f')

  def export_poses(self, output_path, frame_skip=1):
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    print 'exporting', len(self.frames)//frame_skip, 'camera poses to', output_path
    for f in range(0, len(self.frames), frame_skip):
      self.save_mat_to_file(self.frames[f].camera_to_world, os.path.join(output_path, str(f) + '.txt'))

  def export_intrinsics(self, output_path):
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    print 'exporting camera intrinsics to', output_path
    self.save_mat_to_file(self.intrinsic_color, os.path.join(output_path, 'intrinsic_color.txt'))
    self.save_mat_to_file(self.extrinsic_color, os.path.join(output_path, 'extrinsic_color.txt'))
    self.save_mat_to_file(self.intrinsic_depth, os.path.join(output_path, 'intrinsic_depth.txt'))
    self.save_mat_to_file(self.extrinsic_depth, os.path.join(output_path, 'extrinsic_depth.txt'))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('in_path', type=str)
  parser.add_argument('out_path', type=str)
  parser.add_argument('--all_scenes', action='store_true', default=False)
  args = parser.parse_args()

  if not os.path.exists(args.out_path):
      os.makedirs(args.out_path)

  in_path = os.path.join(args.in_path, "scans")

  if args.all_scenes:
    scenes = [x for x in os.listdir(in_path) if x.startswith("scene")]
  else:
      scenes = ['scene0697_02', 'scene0671_00', 'scene0666_00', 'scene0672_00', 'scene0699_00', 'scene0685_01',
                'scene0673_01', 'scene0686_00', 'scene0673_05', 'scene0667_00', 'scene0694_01', 'scene0694_00',
                'scene0700_01', 'scene0693_00', 'scene0681_00', 'scene0679_01', 'scene0664_01', 'scene0665_01',
                'scene0706_00', 'scene0664_02', 'scene0696_02', 'scene0693_01', 'scene0701_02', 'scene0704_01',
                'scene0674_00', 'scene0678_01', 'scene0670_00', 'scene0701_00', 'scene0667_01', 'scene0664_00',
                'scene0678_00', 'scene0697_00', 'scene0683_00', 'scene0688_00', 'scene0698_00', 'scene0705_00',
                'scene0691_00', 'scene0702_02', 'scene0673_00', 'scene0677_01', 'scene0676_01', 'scene0673_04',
                'scene0687_00', 'scene0678_02', 'scene0696_01', 'scene0689_00', 'scene0697_01', 'scene0673_02',
                'scene0672_01', 'scene0685_02', 'scene0700_02', 'scene0677_00', 'scene0671_01', 'scene0696_00',
                'scene0697_03', 'scene0693_02', 'scene0676_00', 'scene0685_00', 'scene0700_00', 'scene0705_01',
                'scene0670_01', 'scene0679_00']

  for scene in tqdm(scenes, "Processed scenes"):
    out_path = os.path.join(args.out_path, scene)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    sd = SensorData(os.path.join(in_path, scene, scene + ".sens"))
    sd.export_depth_images(os.path.join(out_path, 'depth'))
    sd.export_color_images(os.path.join(out_path, 'color'))
    sd.export_poses(os.path.join(out_path, 'pose'))
    sd.export_intrinsics(os.path.join(out_path, 'intrinsic'))
