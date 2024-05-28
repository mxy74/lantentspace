"""This script runs the whole ACE method."""
import shutil
import sys
import os
import numpy as np
import sklearn.metrics as metrics
import torch
from tcav import utils
import tensorflow as tf

import ace_helpers
from ace import ConceptDiscovery
import argparse


def main(args):
    ###### related DIRs on CNS to store results #######
    discovered_concepts_dir = os.path.join(args.working_dir, 'concepts/')
    results_dir = os.path.join(args.working_dir, 'results/')
    cavs_dir = os.path.join(args.working_dir, 'cavs/')
    activations_dir = os.path.join(args.working_dir, 'acts/')
    results_summaries_dir = os.path.join(args.working_dir, 'results_summaries/')
    # if tf.gfile.Exists(args.working_dir):
    #     tf.gfile.DeleteRecursively(args.working_dir)
    # tf.gfile.MakeDirs(args.working_dir)
    # tf.gfile.MakeDirs(discovered_concepts_dir)
    # tf.gfile.MakeDirs(results_dir)
    # tf.gfile.MakeDirs(cavs_dir)
    # tf.gfile.MakeDirs(activations_dir)
    # tf.gfile.MakeDirs(results_summaries_dir)
    if os.path.exists(args.working_dir):
        shutil.rmtree(args.working_dir)
    os.makedirs(args.working_dir, exist_ok=True)
    os.makedirs(discovered_concepts_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(cavs_dir, exist_ok=True)
    os.makedirs(activations_dir, exist_ok=True)
    os.makedirs(results_summaries_dir, exist_ok=True)

    random_concept = 'random_discovery'  # Random concept for statistical testing
    sess = utils.create_session()

    mymodel = ace_helpers.make_model(
        sess, args.model_to_run, args.model_path, args.labels_path)



    # Creating the ConceptDiscovery class instance
    cd = ConceptDiscovery(
        # 训练好的模型，在上边运行概念发现算法
        mymodel,
        # 其中一个类别名
        args.target_class,
        # 随机图像组成的概念
        random_concept,

        args.bottlenecks.split(','),
        sess,
        # 包括某个类别的图像以及random
        args.source_dir,
        # 保存激活的目录
        activations_dir,
        # 保存发现的和随机概念的目录
        cavs_dir,
        # 计算每个概念的cav tcav时使用的随即对应个数
        num_random_exp=args.num_random_exp,
        channel_mean=False,
        # 已发现概念最大图片数量？
        max_imgs=args.max_imgs,
        min_imgs=args.min_imgs,
        num_discovery_imgs=args.max_imgs,
        num_workers=args.num_parallel_workers)
    # Creating the dataset of image patches
    # 加载source_dir/target_class种的图片，每张图片进行分割
    # 得到self.dataset, self.image_numbers, self.patches，每个patch的超像素及对应图片编号（这里需要改一下在50000张图中对应）
    # cd.create_patches(param_dict={'n_segments': [15, 50, 80]})
    cd.create_patches(param_dict={'n_segments': [5, 10, 15]})
    # Saving the concept discovery target class images
    image_dir = os.path.join(discovered_concepts_dir, 'images')
    # tf.gfile.MakeDirs(image_dir)
    os.makedirs(image_dir, exist_ok=True)
    # discovery_images是加载source_dir/target_class的图片
    ace_helpers.save_images(image_dir,
                            (cd.discovery_images * 256).astype(np.uint8))
    # Discovering Concepts
    # 将dataset中的概念输入self.model,根据BOTTLENECKS获得中间层输出,聚类,得到符合条件的概念,每个概念包括对应的概念图,补丁和概念对应的图片
    cd.discover_concepts(method='KM', param_dicts={'n_clusters': 25})

    del cd.dataset  # Free memory
    del cd.image_numbers
    del cd.patches
    del cd.image_number_single_class

    # Save discovered concept images (resized and original sized)
    ace_helpers.save_concepts(cd, discovered_concepts_dir)

    # 计算分数
    # Calculating CAVs and TCAV scores
    # 得到每个概念，原始目标图片，随机图片与random_0/1/2/3....的cav_accuraciess
    # cav_accuraciess = cd.cavs(min_acc=0.0)

    # scores = cd.tcavs(test=False)
    #
    # ace_helpers.save_ace_report(cd, cav_accuraciess, scores,
    #                             results_summaries_dir + 'ace_results.txt')
    # Plot examples of discovered concepts
    for bn in cd.bottlenecks:
        ace_helpers.plot_concepts(cd, bn, 10, address=results_dir)
    # Delete concepts that don't pass statistical testing
    # cd.test_and_remove_concepts(scores)



def parse_arguments(argv):
    """Parses the arguments passed to the run.py script."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str,
                        help='''Directory where the network's classes image folders and random
      concept folders are saved.''', default='./ImageNet')
    parser.add_argument('--working_dir', type=str,
                        help='Directory to save the results.', default='./ACE')
    parser.add_argument('--model_to_run', type=str,
                        help='The name of the model.', default='GoogleNet')
    parser.add_argument('--model_path', type=str,
                        help='Path to model checkpoints.', default='./tensorflow_inception_graph.pb')
    parser.add_argument('--labels_path', type=str,
                        help='Path to model checkpoints.', default='./imagenet_labels.txt')
    parser.add_argument('--target_class', type=str,
                        help='The name of the target class to be interpreted', default='zebra')
    parser.add_argument('--bottlenecks', type=str,
                        help='Names of the target layers of the network (comma separated)',
                        default='mixed4c')
    parser.add_argument('--num_random_exp', type=int,
                        help="Number of random experiments used for statistical testing, etc",
                        default=20)
    parser.add_argument('--max_imgs', type=int,
                        help="Maximum number of images in a discovered concept",
                        default=40)
    parser.add_argument('--min_imgs', type=int,
                        help="Minimum number of images in a discovered concept",
                        default=40)
    parser.add_argument('--num_parallel_workers', type=int,
                        help="Number of parallel jobs.",
                        default=0)
    return parser.parse_args(argv)


if __name__ == '__main__':

    main(parse_arguments(sys.argv[1:]))
