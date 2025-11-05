# CLC2018 color palette and class codes from dataset documentation:
clc_palette = [
  '9e9e9e','dcdcdc','cc99ff','cc66ff','9900ff','ffcc99','ff6600','ff0000','cc3300','ffff00',
  'ccff33','99ff33','66ff00','33cc00','009900','006600','99cc00','cccc00','669900','99ffcc',
  '66cc99','339966','006633','99ffff','66ffff','33cccc','009999','003366','99ccff','6699ff',
  '3366ff','0033cc','000099','9999ff','6666cc','333399','ff99cc','ff66cc','ff3399','cc0066',
  '990033','ffcccc','ff99cc','ff66cc'
]

# Class values 1–44 (sequential order per docs)
clc_vis = {'min': 1, 'max': 44, 'palette': clc_palette}

import ipywidgets as widgets
layout = widgets.Layout(
    overflow='auto',
    border='1px solid gray',
    width='250px',
    height='350px',        # 👈 adjust height here
    padding='5px'
)

legend_dict = {
    "111 Continuous urban fabric": "9e9e9e",
    "112 Discontinuous urban fabric": "dcdcdc",
    "121 Industrial or commercial units": "cc99ff",
    "122 Road and rail networks and associated land": "cc66ff",
    "123 Port areas": "9900ff",
    "124 Airports": "ffcc99",
    "131 Mineral extraction sites": "ff6600",
    "132 Dump sites": "ff0000",
    "133 Construction sites": "cc3300",
    "141 Green urban areas": "ffff00",
    "142 Sport and leisure facilities": "ccff33",
    "211 Non-irrigated arable land": "99ff33",
    "212 Permanently irrigated land": "66ff00",
    "213 Rice fields": "33cc00",
    "221 Vineyards": "009900",
    "222 Fruit trees and berry plantations": "006600",
    "223 Olive groves": "99cc00",
    "231 Pastures": "cccc00",
    "241 Annual crops associated with permanent crops": "669900",
    "242 Complex cultivation patterns": "99ffcc",
    "243 Land principally occupied by agriculture, with significant areas of natural vegetation": "66cc99",
    "244 Agro-forestry areas": "339966",
    "311 Broad-leaved forest": "006633",
    "312 Coniferous forest": "99ffff",
    "313 Mixed forest": "66ffff",
    "321 Natural grasslands": "33cccc",
    "322 Moors and heathland": "009999",
    "323 Sclerophyllous vegetation": "003366",
    "324 Transitional woodland-shrub": "99ccff",
    "331 Beaches, dunes, sands": "6699ff",
    "332 Bare rocks": "3366ff",
    "333 Sparsely vegetated areas": "0033cc",
    "334 Burnt areas": "000099",
    "335 Glaciers and perpetual snow": "9999ff",
    "411 Inland marshes": "6666cc",
    "412 Peat bogs": "333399",
    "421 Salt marshes": "ff99cc",
    "422 Salines": "ff66cc",
    "423 Intertidal flats": "ff3399",
    "511 Water courses": "cc0066",
    "512 Water bodies": "990033",
    "521 Coastal lagoons": "ffcccc",
    "522 Estuaries": "ff99cc",
    "523 Sea and ocean": "ff66cc"
}

codes = [int(k[:3]) for k in legend_dict.keys()]

vis = {'min': 1, 'max': len(codes), 'palette': list(legend_dict.values())}

