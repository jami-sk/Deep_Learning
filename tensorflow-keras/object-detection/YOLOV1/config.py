config = dict()
config['image_size'] = (448, 448)
config['grid_size'] = (7, 7)
config['n_boxes'] = 2
config['classes'] = ['aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'person', 'bird', 'cat', 'cow',
                     'dog', 'horse', 'sheep', 'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tv/monitor']
config['class_map'] = {k: idx for idx, k in enumerate(config['classes'])}
config['n_classes'] = len(config['classes'])
config['output_shape'] = (config['grid_size'][0], config['grid_size'][1], config['n_boxes'] * 5 + config['n_classes'])

