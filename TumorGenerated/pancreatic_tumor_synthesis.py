import random, cv2, elasticdeform, numpy, os
from scipy.ndimage import gaussian_filter

'''
    Statistics
'''
statistics_info = {
    'cyst': {
                # Parameters for Shape Genreation
                'size_histogram_cdf': [0.625866050808314, 0.7806004618937644, 0.8383371824480369, 0.8983833718244804, 0.9284064665127021, 0.9445727482678984, 0.9584295612009238, 0.9699769053117783, 0.9838337182448037, 1.0],
                'size_histogram_cutoffs': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],

                # Parameters for Position Generation
                'offset_z_cdf': [0.06712962962962964, 0.15046296296296297, 0.3055555555555556, 0.4583333333333333, 0.6666666666666666, 0.8078703703703703, 0.9189814814814815, 0.9490740740740741, 0.9953703703703703, 1.0],
                'offset_z_curoffs': [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5],

                # Parameters for Texture Generation
                'intensity_difference_coefficient_a': 0.7569,
                'intensity_difference_coefficient_b': -3.6755,
                'intensity_difference_range': 0.05,
                'intensity_sigma': 2,
            },
    'pdac': {
                # Parameters for Shape Genreation
                'size_histogram_cdf': [0.2620689655172414, 0.5300492610837438, 0.6847290640394089, 0.7980295566502463, 0.8837438423645321, 0.9389162561576355, 0.9645320197044335, 0.9852216748768473, 0.994088669950739, 1.0],
                'size_histogram_cutoffs': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],

                # Parameters for Position Generation
                'offset_z_cdf': [0.0029644268774703555, 0.12944664031620554, 0.3310276679841897, 0.45948616600790515, 0.5464426877470355, 0.6946640316205533, 0.8201581027667985, 0.941699604743083, 0.9881422924901185, 1.0],
                'offset_z_curoffs': [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5],

                # Parameters for Texture Generation
                'intensity_difference_coefficient_a': 0.7898,
                'intensity_difference_coefficient_b': -41.303,
                'intensity_difference_range': 0.05,
                'intensity_sigma': 2,
            },
    }

'''
    Supporting functions for Pancreatic Tumor Synthesis
'''
def generate_prob_function(mask_shape):
    sigma = numpy.random.uniform(3,15)
    # uniform noise generate
    a = numpy.random.uniform(0, 1, size=(mask_shape[0],mask_shape[1],mask_shape[2]))

    # Gaussian filter
    a_2 = gaussian_filter(a, sigma=sigma)

    scale = numpy.random.uniform(0.19, 0.21)
    base = numpy.random.uniform(0.04, 0.06)
    a =  scale * (a_2 - numpy.min(a_2)) / (numpy.max(a_2) - numpy.min(a_2)) + base

    return a

def get_texture(mask_shape):
    # get the prob function
    a = generate_prob_function(mask_shape) 

    # sample once
    random_sample = numpy.random.uniform(0, 1, size=(mask_shape[0],mask_shape[1],mask_shape[2]))

    # if a(x) > random_sample(x), set b(x) = 1
    b = (a > random_sample).astype(float)  # int type can't do Gaussian filter

    # Gaussian filter
    if numpy.random.uniform() < 0.7:
        sigma_b = numpy.random.uniform(3, 5)
    else:
        sigma_b = numpy.random.uniform(5, 8)

    # this takes some time
    b2 = gaussian_filter(b, sigma_b)

    # Scaling and clipping
    u_0 = numpy.random.uniform(0.5, 0.55)
    threshold_mask = b2 > 0.12    # this is for calculte the mean_0.2(b2)
    beta = u_0 / (numpy.sum(b2 * threshold_mask) / threshold_mask.sum())
    Bj = numpy.clip(beta*b2, 0, 1)
    
    return Bj

def random_select(mask):
    z_start, z_end = numpy.where(numpy.any(mask, axis=(0, 1)))[0][[0, -1]]
    z = round(random.uniform(0.0, 1.0) * (z_end - z_start)) + z_start

    organ_mask = mask[..., z]

    # erode the mask (we don't want the edge points)
    kernel = numpy.ones((5,5), dtype=numpy.uint8)
    organ_mask = cv2.erode(organ_mask, kernel, iterations=1)

    coordinates = numpy.argwhere(organ_mask == 1)
    random_index = numpy.random.randint(0, len(coordinates))
    xyz = coordinates[random_index].tolist() # get x,y
    xyz.append(z)
    potential_points = xyz

    return potential_points

def get_ellipsoid(x, y, z):
    # x, y, z is the radius of this ellipsoid in x, y, z direction respectly.
    sh = (4*x, 4*y, 4*z)
    out = numpy.zeros(sh, int)
    aux = numpy.zeros(sh)
    radii = numpy.array([x, y, z])
    com = numpy.array([2*x, 2*y, 2*z])  # center point

    # calculate the ellipsoid 
    bboxl = numpy.floor(com-radii).clip(0,None).astype(int)
    bboxh = (numpy.ceil(com+radii)+1).clip(None, sh).astype(int)
    roi = out[tuple(map(slice,bboxl,bboxh))]
    roiaux = aux[tuple(map(slice,bboxl,bboxh))]
    logrid = *map(numpy.square,numpy.ogrid[tuple(
            map(slice,(bboxl-com)/radii,(bboxh-com-1)/radii,1j*(bboxh-bboxl)))]),
    dst = (1-sum(logrid)).clip(0,None)
    mask = dst>roiaux
    roi[mask] = 1
    numpy.copyto(roiaux,dst,where=mask)
    
    return out

def get_bounds_from_cdf(random_number, cdf, cutoff):
    index = 0
    for i in range(len(cdf)-1):
        if cdf[i]<=random_number and random_number<cdf[i+1]:
            index = i
            break
    
    bound_low = cutoff[index]
    bound_high = cutoff[index+1]

    return bound_low, bound_high

def bounding_box_calculation(mask):
    positions = numpy.where(mask>0)
    bbox = [
        [numpy.min(positions[0]), numpy.max(positions[0])], 
        [numpy.min(positions[1]), numpy.max(positions[1])], 
        [numpy.min(positions[2]), numpy.max(positions[2])], 
    ]

    return bbox

'''
    Shape Generation
'''
def shape_generation(mask, tumor_type, center_bbox_tumor, metadata):
    size_low, size_high = get_bounds_from_cdf(
        random_number = random.random(),
        cdf = statistics_info[tumor_type]['size_histogram_cdf'], 
        cutoff = statistics_info[tumor_type]['size_histogram_cutoffs'],
        )
    
    radius_pancreas_x = (metadata['bbox_pancreas'][0][1]-metadata['bbox_pancreas'][0][0])/2.0
    radius_pancreas_y = (metadata['bbox_pancreas'][1][1]-metadata['bbox_pancreas'][1][0])/2.0
    radius_pancreas_z = (metadata['bbox_pancreas'][2][1]-metadata['bbox_pancreas'][2][0])/2.0

    x = random.randint(int(size_low*radius_pancreas_x), int(size_high*radius_pancreas_x))
    y = random.randint(int(size_low*radius_pancreas_y), int(size_high*radius_pancreas_y))
    z = random.randint(int(size_low*radius_pancreas_z), int(size_high*radius_pancreas_z))
    
    geo = get_ellipsoid(x, y, z)

    sigma = random.randint(1, 2)
    geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
    geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
    geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))

    geo_mask = numpy.zeros((
        mask.shape[0] + metadata['enlarge'][0], 
        mask.shape[1] + metadata['enlarge'][1],
        mask.shape[2] + metadata['enlarge'][2]), 
        dtype=numpy.int8)
    
    geo_mask[
        center_bbox_tumor[0]-geo.shape[0]//2:center_bbox_tumor[0]+geo.shape[0]//2,
        center_bbox_tumor[1]-geo.shape[1]//2:center_bbox_tumor[1]+geo.shape[1]//2,
        center_bbox_tumor[2]-geo.shape[2]//2:center_bbox_tumor[2]+geo.shape[2]//2,
        ] += geo

    # as long as we enlarge out sapce before, we need to cut it back
    geo_mask = geo_mask[
        metadata['enlarge'][0]//2:-metadata['enlarge'][0]//2, 
        metadata['enlarge'][1]//2:-metadata['enlarge'][1]//2, 
        metadata['enlarge'][2]//2:-metadata['enlarge'][2]//2,
        ]
    geo_mask = (geo_mask * mask) >=1

    return geo_mask

'''
    Position Generation
'''
def position_generation(mask, tumor_type):
    bbox_pancreas = bounding_box_calculation(mask)
    pancreas_height = bbox_pancreas[2][1] - bbox_pancreas[2][0]
    center_bbox_pancreas_z = (bbox_pancreas[2][1] + bbox_pancreas[2][0])/2.0
    
    # we need to enlarge the sample space to avoid boundary check (which will be very annoying)
    # by enlarge the space, all we need to do is change the place point.
    enlarge_x, enlarge_y, enlarge_z = 150, 150, int(1.3*pancreas_height)
        
    offset_low, offset_high = get_bounds_from_cdf(
        random_number = random.random(),
        cdf = statistics_info[tumor_type]['offset_z_cdf'], 
        cutoff = statistics_info[tumor_type]['offset_z_curoffs'],
        )

    while True:
        center_bbox_tumor = random_select(mask)
        offset_ratio = (center_bbox_tumor[2]-center_bbox_pancreas_z)/pancreas_height
        if offset_low<=offset_ratio and offset_ratio<=offset_high:
            break

    center_bbox_tumor = [center_bbox_tumor[0] + enlarge_x//2, center_bbox_tumor[1] + enlarge_y//2, center_bbox_tumor[2] + enlarge_z//2]

    metadata = {
        'enlarge': [enlarge_x, enlarge_y, enlarge_z],
        'pancreas_height': pancreas_height,
        'bbox_pancreas': bbox_pancreas,
    }
    return center_bbox_tumor, metadata

'''
    Texture Generation
'''
def texture_generation(image, mask, tumor_type, mask_generated):
    texture = get_texture(mask.shape)

    sigma = numpy.random.uniform(1, statistics_info[tumor_type]['intensity_sigma'])
    median_healthy = numpy.median(image[numpy.where((mask_generated>0) & (mask>0))])
    median_target = statistics_info[tumor_type]['intensity_difference_coefficient_a'] * median_healthy + statistics_info[tumor_type]['intensity_difference_coefficient_b']
    range = statistics_info[tumor_type]['intensity_difference_range']
    difference = median_healthy - median_target
    difference = numpy.random.uniform(int((1-range)*difference), int((1+range)*difference))

    # blur the boundary
    geo_blur = gaussian_filter(mask_generated*255, sigma)
    abnormally = (image - texture * (geo_blur/255) * difference) * mask_generated
    
    image = image * (1 - mask_generated) + abnormally
    mask = mask + mask_generated
    return image, mask

'''
    Entry Point Function
'''
def synthesize_pancreatic_tumor(image, mask, tumor_type):
    center_bbox_tumor, metadata = position_generation(mask, tumor_type)
    mask_generated = shape_generation(mask, tumor_type, center_bbox_tumor, metadata)
    image, mask = texture_generation(image, mask, tumor_type, mask_generated)
    return image, mask

'''
image     : 3d numpy array
mask      : same shape with image
tumor type: "pdac" or "cyst"
'''