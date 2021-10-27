__all__ = ['config']

config = dict()
config['image_size'] = (448, 448)
config['grid_size'] = (7, 7)
config['n_boxes'] = 2
config['classes'] = ['aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'person', 'bird', 'cat', 'cow',
                     'dog', 'horse', 'sheep', 'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tv/monitor']
config['class_map'] = {k: idx for idx, k in enumerate(config['classes'])}
config['n_classes'] = len(config['classes'])
config['output_shape'] = (config['grid_size'][0], config['grid_size'][1], config['n_boxes'] * 5 + config['n_classes'])
config['model_arch'] = [  # tuple : (block, kernel_size, filters, stride, padding)
    # tuple : (block, pool_size, stride_size, padding)
    ('CNN', 7, 64, 2, 'same'),
    ('Max', 2, 2, 'same'),
    ('CNN', 3, 192, 1, 'same'),
    ('Max', 2, 2, 'same'),
    ('CNN', 1, 128, 1, 'same'),
    ('CNN', 3, 256, 1, 'same'),
    ('CNN', 1, 256, 1, 'same'),
    ('CNN', 3, 512, 1, 'same'),
    ('Max', 2, 2, 'same'),
    ('CNN', 1, 256, 1, 'same'),
    ('CNN', 3, 512, 1, 'same'),
    ('CNN', 1, 256, 1, 'same'),
    ('CNN', 3, 512, 1, 'same'),
    ('CNN', 1, 256, 1, 'same'),
    ('CNN', 3, 512, 1, 'same'),
    ('CNN', 1, 256, 1, 'same'),
    ('CNN', 3, 512, 1, 'same'),
    ('CNN', 1, 512, 1, 'same'),
    ('CNN', 3, 1024, 1, 'same'),
    ('Max', 2, 2, 'same'),
    ('CNN', 1, 512, 1, 'same'),
    ('CNN', 3, 1024, 1, 'same'),
    ('CNN', 1, 512, 1, 'same'),
    ('CNN', 3, 1024, 1, 'same'),
    ('CNN', 3, 1024, 1, 'same'),
    ('CNN', 3, 1024, 2, 'same'),
    ('CNN', 3, 1024, 1, 'same'),
    ('CNN', 3, 1024, 1, 'same')
]

