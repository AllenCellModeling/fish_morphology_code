import pandas as pd
from quilt3distribute import Dataset

df = pd.read_csv('AssayDevFishAnalsysis2020-IntensityFeatures.csv')

metadata = [
	{'CellId': {'name': None,
		'description': 'Unique id that indentifies the FOV and the cell mask label'}},
	{'napariCell_ObjectNumber': {'name': None,
		'description': 'Unique id that indentifies the label of cell segmentation in the fov'}},
	# {'Score1': {'name': None,
	# 	'description': 'Structure organization by scorer 1'}},
	# {'Score2': {'name': None,
	# 	'description': 'Structure organization by scorer 2'}},
	{'Age': {'name': None,
		'description': 'Cells age'}},
	{'ImagePath': {'name': None,
		'description': 'Z Stack with data produced by assay-dev'}},
	{'original_fov_location': {'name': 'original_fov_location',
		'description': 'Path to raw data'}},
	{'TotalArea': {'name': None,
		'description': 'Number of pixels in cell mask'}},
	{'FracAreaBackground': {'name': None,
		'description': 'Fraction of cell area classified as background'}},
	{'FracAreaMessy': {'name': None,
		'description': 'Fraction of cell area classified as messy'}},
	{'FracAreaThreads': {'name': None,
		'description': 'Fraction of cell area classified as threads'}},
	{'FracAreaRandom': {'name': None,
		'description': 'Fraction of cell area classified as random dots'}},
	{'FracAreaRegularDots': {'name': None,
		'description': 'Fraction of cell area classified as regular dots'}},
	{'FracAreaRegularStripes': {'name': None,
		'description': 'Fraction of cell area classified as regular stripes'}},
	{'ProbMessy': {'name': None,
		'description': 'Average probability of a pixel inside the cell to be classified as messy'}},
	{'ProbThreads': {'name': None,
		'description': 'Average probability of a pixel inside the cell to be classified as threads'}},
	{'ProbRandom': {'name': None,
		'description': 'Average probability of a pixel inside the cell to be classified as random'}},
	{'ProbRegularDots': {'name': None,
		'description': 'Average probability of a pixel inside the cell to be classified as regular dots'}},
	{'ProbRegularStripes': {'name': None,
		'description': 'Average probability of a pixel inside the cell to be classified as regular stripes'}},
	# {'SarcomereWidth': {'name': None,
	# 	'description': 'The average major axis length of the Voronoi units classified as regular stripes'}},
	# {'SarcomereLength': {'name': None,
	# 	'description': 'The average minor axis length of the Voronoi units classified as regular stripes'}},
	# {'NRegularStripesVoronoi': {'name': None,
	# 	'description': 'Number of Voronoi units classified as regular stripes used to calculate length and width'}},
	# {'RadonDominantAngleEntropyThreads': {'name': None,
	# 	'description': 'Entropy of dominant angle averaged over areas classified and threads'}},
	# {'RadonResponseRatioAvgThreads': {'name': None,
	# 	'description': 'Radon response ratio averaged over areas classified as threads'}},
	# {'RadonDominantAngleEntropyRegStripes': {'name': None,
	# 	'description': 'Entropy of dominant angle averaged over areas classified and regular stripes'}},
	# {'RadonResponseRatioAvgRegStripes': {'name': None,
	# 	'description': 'Radon response ratio averaged over areas classified as regular stripes'}}
	{'Intensity_Median': {'name': 'IntensityMedian',
		'description': 'Median of GFP signal in cell mask'}},
	{'Intensity_Integrated': {'name': 'IntensityIntegrated',
		'description': 'Integrated GFP signal in cell mask'}},
	{'Intensity_Median_BackSub': {'name': 'IntensityMedianBkgSub',
		'description': 'Median of GFP signal in cell mask with background subtracted (10% percentile'}},
	{'Intensity_Integrated_BackSub': {'name': 'IntensityIntegratedBkgSub',
		'description': 'Integrated GFP signal in cell mask with background subtracted (10% percentile'}},
	{'MaxCoeffVar': {'name': None,
		'description': 'Maximum value of the coefficient of variation obtained from correlation plots'}},
	{'HPeak': {'name': None,
		'description': 'High of the highest peak in the correlation plots'}},
	{'PeakDistance': {'name': None,
		'description': 'Distance in pixels in which the maximum of the highest peak occurs'}},
	{'PeakAngle': {'name': None,
		'description': 'Angle in degrees for which we observe the highest correlation value'}}
]

selected_features = [key for f in metadata for key,_ in f.items()]

df = df[selected_features]

features_rename = [{key: value['name']} for f in metadata for key,value in f.items() if value['name'] is not None]

for feature in features_rename:
	df = df.rename(columns=feature)

with open('assay-dev-fish.md','w') as ftxt:
	for meta in metadata:
		for key, value in meta.items():
			ftxt.write("Feature: {0}, Description: {1}\n".format(
				value['name'] if value['name'] is not None else key,
				value['description']))

df.to_csv('AssayDevFishAnalsysis2019-Handoff-V5.csv')

if True:

	# Create the dataset
	ds = Dataset(
	    dataset="AssayDevFishAnalsysis2019-Handoff-V5.csv",
	    name="assay_dev_fish_analysis",
	    package_owner="matheus",
	    readme_path="assay-dev-fish.md"
	)

	# Optionally add common additional requirements
	# ds.add_usage_doc("https://docs.quiltdata.com/walkthrough/reading-from-a-package")
	# ds.add_license("https://www.allencell.org/terms-of-use.html")

	# Optionally indicate column values to use for file metadata
	ds.set_metadata_columns(["CellId"])
	ds.set_path_columns(
		['ImagePath']
	)
	# Optionally rename the columns on the package level
	# ds.set_column_names_map({
	#     "2dReadPath": "images_2d",
	#     "3dReadPath": "images_3d"
	# })

	# Distribute
	pkg = ds.distribute(push_uri="s3://allencell-internal-quilt", message="Fish dataset by assay-dev")