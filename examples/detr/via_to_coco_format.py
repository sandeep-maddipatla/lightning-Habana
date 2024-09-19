import convert as via2coco

# Run this from ViaCoCo checkout. Instructions below
#
# !git clone https://github.com/woctezuma/VIA2COCO
# %cd VIA2COCO/
# !git checkout fixes
#
# PYTHONPATH=${PYTHONPATH}:$(pwd) python /path/to/detr-ft/via_to_coco_format.py

data_path = '/root/gs-274/balloon/'

first_class_index = 0

for keyword in ['train', 'val']:

    input_dir = data_path + keyword + '/'
    input_json = input_dir + 'via_region_data.json'
    categories = ['balloon']
    super_categories = ['N/A']
    output_json = input_dir + 'custom_' + keyword + '.json'

    print('Converting {} from VIA format to COCO format'.format(input_json))

    coco_dict = via2coco.convert(
        imgdir=input_dir,
        annpath=input_json,
        categories=categories,
        super_categories=super_categories,
        output_file_name=output_json,
        first_class_index=first_class_index,
    )
