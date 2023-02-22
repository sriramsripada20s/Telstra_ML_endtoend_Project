import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import streamlit as st
from PIL import Image

pickle_in = open("lgbm_classifier.pkl","rb")
classifier = pickle.load(pickle_in)

st.title("Telstra Fault Severity Prediction")

st.image('telstra.png')

st.write("""

This is a Streamlit web app created so users could explore my LGBM Classification model predicting the
Telstra network's Fault Severity at a time at a particular location based on the log data available.

The dataset can be found [here](https://www.kaggle.com/c/telstra-recruiting-network/data) 

Fault severity has 3 categories: 0,1,2 (0 meaning no fault, 1 meaning only a few, and 2 meaning many). 
""")


def visualize_confidence_level(prediction_proba):
    """
    this function uses matplotlib to create inference bar chart rendered with streamlit in real-time
    return type : matplotlib bar chart
    """
    data = (prediction_proba[0] * 100).round(2)
    grad_percentage = pd.DataFrame(data=data, columns=['Percentage'], index=['Fault Sev0', 'Fault Sev1', 'Fault Sev2'])
    ax = grad_percentage.plot(kind='barh', figsize=(7, 4), color='#FF2800', zorder=10, width=0.5)
    ax.legend().set_visible(False)
    ax.set_xlim(xmin=0, xmax=100)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off",
                   labelleft="on")

    vals = ax.get_xticks()
    for tick in vals:
        ax.axvline(x=tick, linestyle='dashed', alpha=0.6, color='#eeeeee', zorder=1)

    ax.set_xlabel(" Percentage(%) Confidence Level", labelpad=2, weight='bold', size=12)
    ax.set_ylabel("Fault Severity Types", labelpad=10, weight='bold', size=12)
    ax.set_title('Prediction Confidence Level ', fontdict=None, loc='center', pad=None, weight='bold')

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(bbox_inches='tight')
    return

st.sidebar.header('User Input Features')

def get_user_input():
    """
    this function is used to get user input using sidebar slider and selectbox
    return type : pandas dataframe
    """

    location_id = st.sidebar.selectbox("Select Location ID", (118, 91, 152, 931, 120, 664, 640, 122, 263, 613, 760, 519, 746, 1066, 812, 343, 133, 976, 948, 808, 894, 875, 1024, 166, 687, 1016, 707, 978, 775, 829, 732, 508, 257, 116, 830, 491, 253, 740, 704, 1089, 653, 600, 892, 802, 794, 477, 684, 895, 496, 321, 1107, 1086, 1099, 975, 1019, 342, 744, 181, 1052, 696, 1008, 798, 777, 95, 699, 272, 846, 795, 826, 112, 778, 810, 690, 494, 1095, 445, 478, 661, 380, 242, 1082, 1007, 131, 980, 373, 169, 421, 779, 59, 648, 124, 7, 12, 1026, 270, 1106, 1090, 522, 836, 145, 1020, 804, 1055, 157, 893, 457, 99, 839, 719, 793, 1109, 957, 665, 646, 302, 273, 1, 476, 734, 1076, 520, 1046, 803, 963, 821, 1061, 714, 300, 611, 724, 173, 943, 733, 249, 408, 723, 255, 655, 888, 989, 1015, 692, 945, 651, 763, 416, 497, 890, 849, 208, 139, 914, 326, 937, 506, 149, 865, 925, 243, 193, 471, 375, 1047, 490, 1051, 896, 1011, 834, 97, 226, 962, 885, 816, 638, 468, 1122, 1098, 376, 303, 126, 507, 709, 466, 1102, 998, 727, 76, 309, 737, 1111, 314, 1023, 658, 990, 618, 758, 488, 643, 607, 141, 332, 814, 921, 742, 897, 240, 691, 809, 549, 266, 1010, 1094, 345, 501, 939, 823, 244, 857, 1065, 906, 404, 1117, 461, 1056, 995, 500, 100, 311, 390, 756, 90, 479, 1025, 473, 161, 1029, 711, 1054, 974, 1115, 867, 686, 899, 202, 923, 771, 485, 919, 1009, 973, 504, 1058, 1084, 234, 418, 107, 878, 444, 599, 155, 717, 991, 1042, 35, 465, 499, 597, 101, 864, 845, 695, 738, 1063, 407, 984, 641, 644, 276, 1075, 252, 1083, 972, 1048, 1100, 324, 847, 484, 230, 9, 1049, 229, 322, 148, 57, 462, 790, 662, 679, 533, 51, 1125, 135, 297, 498, 526, 603, 619, 805, 159, 108, 765, 609, 584, 455, 856, 1093, 154, 102, 493, 891, 278, 130, 398, 754, 926, 681, 62, 442, 807, 282, 337, 822, 495, 1050, 236, 934, 534, 420, 645, 13, 700, 400, 128, 983, 682, 918, 964, 288, 175, 328, 147, 518, 879, 1037, 262, 296, 999, 767, 721, 546, 171, 281, 1103, 469, 1119, 344, 876, 635, 1014, 1104, 480, 627, 298, 755, 642, 363, 451, 735, 800, 902, 968, 944, 505, 26, 369, 1031, 903, 459, 979, 335, 115, 379, 1072, 1030, 819, 1038, 1044, 837, 1096, 49, 848, 1034, 745, 206, 930, 446, 475, 467, 1022, 292, 448, 460, 46, 938, 601, 884, 1118, 307, 942, 489, 509, 863, 441, 932, 280, 88, 1000, 333, 782, 712, 1120, 946, 1005, 301, 1018, 558, 409, 349, 470, 358, 951, 68, 596, 415, 44, 1078, 674, 673, 774, 378, 104, 936, 34, 463, 818, 786, 693, 909, 967, 515, 151, 994, 481, 8, 256, 881, 140, 267, 1080, 424, 844, 710, 628, 768, 277, 325, 977, 842, 1032, 1067, 838, 860, 949, 239, 827, 912, 851, 137, 573, 924, 866, 313, 123, 667, 22, 532, 706, 382, 381, 1053, 916, 83, 67, 283, 886, 1045, 20, 1079, 624, 853, 722, 535, 305, 374, 820, 563, 560, 385, 251, 89, 1036, 114, 56, 450, 72, 955, 889, 197, 748, 3, 2, 248, 350, 512, 371, 1041, 474, 403, 247, 825, 167, 616, 81, 436, 211, 228, 365, 764, 223, 284, 1087, 163, 36, 566, 652, 622, 472, 360, 981, 1021, 127, 759, 287, 632, 904, 1064, 268, 245, 92, 1097, 869, 576, 817, 1081, 235, 762, 602, 413, 23, 103, 384, 588, 1017, 1062, 1091, 17, 702, 880, 93, 406, 578, 447, 106, 347, 16, 78, 649, 125, 362, 182, 330, 264, 395, 172, 290, 188, 195, 961, 659, 73, 969, 663, 218, 19, 773, 1092, 27, 871, 523, 672, 769, 1069, 579, 797, 657, 32, 353, 783, 410, 615, 286, 110, 557, 788, 757, 388, 666, 260, 204, 583, 318, 79, 437, 1112, 1121, 323, 179, 405, 435, 426, 697, 604, 1033, 877, 527, 941, 74, 933, 521, 544, 726, 259, 528, 731, 39, 136, 30, 210, 213, 274, 1070, 529, 850, 806, 246, 221, 425, 299, 996, 440, 483, 399, 232, 285, 741, 184, 414, 1006, 392, 367, 320, 831, 429, 950, 258, 352, 1113, 37, 294, 789, 592, 165, 38, 162, 883, 1060, 389, 432, 453, 559, 680, 143, 1116, 705, 33, 555, 170, 15, 736, 540, 241, 334, 430, 715, 516, 354, 511, 231, 198, 813, 119, 589, 393, 1035, 419, 443, 117, 1088, 785, 716, 617, 1108, 177, 113, 882, 543, 1126, 144, 874, 610, 550, 423, 772, 417, 531, 1105, 47, 394, 841, 541, 10, 637, 536, 346, 82, 922, 189, 537, 799, 146, 50, 887, 959, 153, 237, 811, 1074, 694, 547, 331, 304, 514, 1073, 562, 840, 670, 254, 766, 310, 953, 338, 356, 348, 402, 629, 1027, 222, 250, 815, 357, 855, 289, 156, 427, 752, 1101, 317, 792, 45, 396, 928, 551, 194, 315, 621, 608, 336, 654, 713, 158, 525, 870, 1110, 183, 548, 65, 556, 872, 577, 359, 4, 570, 63, 386, 401, 971, 587, 312, 205, 1059, 660, 854, 966, 5, 614, 960, 565, 150, 524, 595, 200, 186, 582, 269, 987, 675, 339, 355, 238, 434, 639, 439, 196, 708, 190, 901, 503, 215, 86, 929, 917, 464, 671, 14, 633, 796, 214, 625, 6, 898, 87, 852, 1013, 956, 1071, 1077, 678, 370, 224, 1068, 180, 958, 824, 1002, 487, 492, 271, 905, 911, 225, 718, 60, 683, 676, 568, 915, 668, 84, 291, 530, 538, 21, 203, 53, 187, 293, 319, 564, 623, 859, 199, 391, 835, 329, 947, 220, 85, 18, 561))
    event_type = st.sidebar.selectbox("Select Event Type", (69, 26, 101, 11, 50, 13, 31, 24, 34, 25, 174, 35, 15, 20, 129, 23, 68, 21, 65, 54, 55, 47, 80, 86, 136, 33, 89, 179, 42, 38, 57, 60, 131, 29, 48, 90, 51, 56, 224, 46, 41, 22, 43, 81, 93, 66, 67, 76, 393, 9, 113, 10, 18, 32, 19, 75, 52, 77, 36, 73, 30, 78, 97, 70, 124, 28, 143, 45, 105, 27, 100, 49, 59, 40, 138, 74, 103, 96, 92, 116, 91, 14, 37, 5, 61, 53, 44, 265, 79, 266, 84, 85, 87, 82, 72, 64, 142, 126, 119, 17, 63, 199, 368, 95, 16, 108, 110, 99, 102, 39, 104, 161, 271))
    resource_type = st.sidebar.selectbox("Select Resource Type", (2, 8, 10, 14, 12, 7, 21, 11, 15, 24, 18, 6, 9, 13, 25, 19, 17, 27, 20, 3, 16, 1, 35, 5, 31, 22))
    severity_type = st.sidebar.selectbox("Select Severity Type", (2, 1, 4, 5, 3))
    volume = st.sidebar.slider('volume', 1, 1649, 10)
    feature_type = st.sidebar.selectbox("Select Feature Type", (544, 550, 522, 365, 1536, 226, 374, 1640, 153, 432, 569, 341, 444, 459, 1090, 163, 264, 2654, 283, 73, 312, 54, 224, 281, 285, 2257, 353, 364, 1624, 268, 836, 292, 144, 1983, 306, 869, 1948, 356, 851, 858, 411, 625, 1927, 360, 654, 566, 319, 1618, 929, 162, 549, 574, 2007, 77, 2700, 380, 294, 1286, 232, 1078, 1505, 636, 686, 1632, 2174, 770, 74, 313, 1783, 988, 1093, 940, 1630, 2228, 1956, 134, 467, 171, 82, 638, 238, 301, 68, 71, 2166, 346, 228, 377, 865, 1945, 1405, 468, 823, 152, 498, 1100, 345, 509, 296, 763, 430, 355, 270, 199, 182, 1273, 857, 1586, 659, 4119, 932, 548, 87, 1058, 546, 55, 619, 747, 170, 1988, 1122, 261, 219, 207, 875, 3391, 246, 65, 830, 42, 559, 349, 161, 1086, 1092, 1035, 385, 928, 233, 650, 433, 683, 515, 423, 478, 517, 748, 1096, 66, 511, 1076, 497, 861, 1715, 2143, 3567, 1172, 899, 254, 2016, 920, 2747, 1334, 1126, 2738, 656, 195, 70, 3231, 1397, 800, 2716, 1925, 1094, 794, 2085, 520, 483, 753, 974, 1620, 557, 889, 274, 534, 334, 1166, 536, 629, 1051, 1443, 1308, 692, 2712, 897, 591, 785, 327, 637, 1888, 572, 1616, 1634, 1952, 626, 372, 62, 2599, 201, 1318, 435, 339, 197, 204, 80, 239, 344, 2067, 837, 3580, 363, 51, 1947, 381, 2722, 440, 4687, 779, 603, 466, 1176, 2357, 2128, 722, 1391, 1007, 761, 2172, 4890, 463, 15, 1164, 1043, 1389, 257, 3278, 56, 1953, 210, 2199, 583, 2715, 486, 555, 347, 846, 1368, 709, 538, 1401, 1693, 1615, 63, 1628, 663, 1982, 2188, 1329, 1157, 2168, 744, 2691, 1066, 395, 315, 2245, 450, 154, 3023, 202, 845, 2720, 628, 966, 421, 634, 1198, 273, 2369, 272, 316, 833, 1732, 860, 280, 1419, 2476, 426, 1119, 545, 2293, 302, 3871, 172, 1259, 240, 81, 910, 1717, 3804, 1082, 739, 718, 551, 2364, 191, 529, 938, 516, 1247, 1214, 821, 504, 4658, 859, 657, 209, 919, 2176, 1587, 1420, 781, 999, 586, 1325, 263, 540, 853, 244, 3043, 1626, 1552, 1479, 771, 4498, 1398, 593, 590, 122, 997, 351, 2182, 1408, 1483, 1149, 777, 225, 827, 734, 399, 2272, 513, 487, 885, 1095, 3878, 412, 1222, 1622, 418, 1493, 695, 883, 1465, 577, 4896, 75, 183, 434, 3046, 618, 352, 359, 531, 2177, 972, 1986, 337, 1601, 222, 218, 842, 37, 236, 384, 141, 1034, 3337, 457, 4101, 2499, 413, 3591, 278, 44, 615, 948, 105, 926, 2486, 156, 3600, 725, 124, 856, 1071, 783, 646, 1127, 750, 1121, 740, 1163, 751, 1644, 160, 579, 3339, 595, 420, 525, 678, 1202, 2493, 40, 1411, 1105, 368, 139, 1319, 2692, 328, 819, 1228, 1246, 39, 3242, 1923, 3250, 510, 1185, 5203, 1131, 1032, 1080, 181, 863, 2501, 592, 713, 1482, 712, 1232, 1054, 3189, 704, 1028, 752, 2117, 635, 690, 1307, 1200, 581, 754, 766, 494, 53, 518, 52, 206, 696, 1063, 265, 3254, 1023, 392, 2706, 408, 852, 3875, 4661, 2796, 1697, 720, 1089, 2249, 10, 47, 1859, 2542, 1367, 1327, 1404, 2071, 2479, 3344, 627, 1375, 964, 578, 1407, 1282, 677, 193, 2286, 946, 1826, 323, 942, 2243, 1328, 2180, 1579, 1130, 4665, 247, 2162, 472, 234, 1376, 484, 4115, 612, 698, 716, 888, 700, 1771, 479, 2051, 614, 774, 2422, 679, 436, 3802, 708, 1216, 3118, 94, 565, 1218, 401, 1041, 1553, 3021, 850, 711, 505, 983, 1818, 849, 1084, 1705, 458, 482, 2717, 4109, 588, 2718, 1000, 562, 564, 167, 2202, 2804, 376, 4733, 1692, 970, 714, 547, 69, 825, 1703, 8, 1385, 1320, 1946, 220, 613, 847, 1714, 2710, 2278, 1152, 489, 1394, 3019, 216, 1276, 76, 371, 279, 1939, 314, 2636, 1251, 386, 968, 669, 1097, 2146, 3044, 237, 342, 1048, 1606, 277, 924, 2109, 915, 1544, 617, 2979, 775, 573, 1268, 309, 417, 325, 996, 2371, 934, 2261, 616, 1162, 524, 1312, 50, 874, 1738, 2943, 789, 85, 2726, 2981, 3568, 1642, 317, 1989, 1047, 379, 336, 824, 133, 2000, 138, 1188, 1915, 602, 576, 941, 1169, 373, 911, 293, 348, 1271, 1013, 241, 930, 431, 2125, 1254, 1943, 651, 449, 543, 415, 882, 252, 2492, 3577, 2473, 813, 4429, 1470, 3892, 2158, 3330, 2505, 1875, 2352, 1646, 212, 3579, 699, 810, 530, 817, 3796, 1429, 2483, 782, 115, 1864, 541, 665, 922, 2192, 235, 1024, 1012, 338, 501, 537, 1492, 1608, 3264, 2740, 145, 3340, 1002, 1698, 3030, 1033, 481, 575, 4968, 980, 1137, 784, 1275, 419, 1638, 1665, 2154, 2280, 136, 1288, 3575, 1009, 764, 503, 159, 188, 1937, 622, 262, 198, 937, 898, 1245, 4116, 532, 957, 1402, 844, 2340, 1809, 765, 147, 649, 623, 1941, 1856, 297, 2170, 1706, 2515, 4107, 151, 402, 2355, 710, 528, 242, 1015, 2564, 2567, 1236, 135, 1513, 870, 1314, 2477, 86, 4583, 507, 933, 601, 891, 1730, 902, 2556, 403, 2027, 1148, 3332, 795, 1377, 841, 1563, 1239, 389, 375, 2149, 350, 2218, 1098, 362, 1660, 982, 289, 1173, 2330, 820, 567, 205, 597, 196, 1803, 1147, 1934, 1599, 1285, 1897, 1369, 1409, 3553, 681, 129, 310, 1190, 1445, 866, 2704, 1170, 979, 1165, 936, 887, 4653, 960, 1556, 832, 1473, 2576, 259, 523, 305, 2803, 208, 2390, 4346, 1467, 519, 621, 703, 1344, 1399, 1174, 2791, 835, 3501, 2088, 3806, 499, 132, 1160, 1113, 492, 2248, 442, 2251, 391, 561, 1507, 812, 609, 1722, 2498, 303, 128, 1931, 46, 1427, 333, 1436, 769, 393, 2164, 1213, 2253, 1938, 387, 2356, 311, 1310, 580, 2475, 943, 604, 2491, 1211, 1439, 585, 230, 3121, 862, 2095, 192, 804, 892, 1010, 1194, 1498, 61, 3879, 1527, 1600, 1065, 772, 1309, 717, 343, 1373, 3961, 2418, 1430, 3964, 563, 4336, 992, 780, 829, 798, 944, 606, 1108, 905, 2004, 1104, 269, 2782, 1763, 1053, 1957, 455, 3353, 640, 1026, 471, 4425, 4734, 286, 2730, 2879, 912, 917, 3029, 571, 211, 1341, 2083, 1455, 630, 4190, 3352))
    log_volumecount = st.sidebar.slider('log_volumecount', 1, 19, 1)
    log_volumemin = st.sidebar.slider('log_volumemin', 1, 354, 10)
    log_volumemean = st.sidebar.slider('log_volumemean', 1, 412, 10)
    log_volumemax = st.sidebar.slider('log_volumemax', 1, 877, 10)
    log_volumestd = st.sidebar.slider('log_volumestd', 0, 423, 10)
    log_volumesum = st.sidebar.slider('log_volumesum', 1, 1649, 10)
    event_volumecount = st.sidebar.slider('event_volumecount', 1, 135, 4)
    event_volumemin = st.sidebar.slider('event_volumemin ', 1, 354, 1)
    event_volumemean = st.sidebar.slider('event_volumemean', 1, 412, 1)
    event_volumemax = st.sidebar.slider('event_volumemax', 1, 877, 1)
    event_volumestd = st.sidebar.slider('event_volumestd', 0, 10, 1)
    event_volumesum = st.sidebar.slider('event_volumesum', 1, 50, 1)
    sev_volumecount = st.sidebar.slider('sev_volumecount', 1, 20, 1)
    sev_volumemin = st.sidebar.slider('sev_volumemin', 1, 430, 1)
    sev_volumemean = st.sidebar.slider('sev_volumemean', 1, 10, 1)
    sev_volumemax = st.sidebar.slider('sev_volumemax', 1, 15, 1)
    sev_volumestd = st.sidebar.slider('sev_volumestd', 0, 5, 1)
    sev_volumesum = st.sidebar.slider('sev_volumesum', 1, 25, 1)
    log_volume = st.sidebar.slider('log_volume', 0, 8, 1)
    avgvol_per_loc = st.sidebar.slider('avgvol_per_loc', 0, 6, 1)
    maxvol_per_loc = st.sidebar.slider('maxvol_per_loc', 0, 8, 1)
    minvol_per_loc = st.sidebar.slider('minvol_per_loc', 0.0, 1.0, 0.1)
    medianvol_per_loc = st.sidebar.slider('medianvol_per_loc', 0, 4, 1)
    stdvol_per_loc = st.sidebar.slider('stdvol_per_loc', 0.0, 2.0, 0.2)
    order = st.sidebar.slider('order', 0.0, 1.0, 0.1)
    event_order = st.sidebar.slider('event_order', 0.0, 1.0, 0.1)
    feature_order = st.sidebar.slider('feature_order', 0.0, 1.0, 0.1)
    location_rank_asc = st.sidebar.slider('location_rank_asc', 1, 30, 1)
    location_rank_desc = st.sidebar.slider('location_rank_desc', 1, 30, 1)
    location_count = st.sidebar.slider('location_count', 1, 56, 2)
    loc_rank_rel = st.sidebar.slider('location_rank_rel', 0.0, 1.0, 0.1)


    features = {'location_id': location_id,
                'event_type': event_type,
                'resource_type': resource_type,
                'severity_type': severity_type,
                'volume': volume,
                'feature_type': feature_type,
                'log_volumecount': log_volumecount,
                'log_volumemin': log_volumemin,
                'log_volumemean': log_volumemean,
                'log_volumemax': log_volumemax,
                'log_volumestd': log_volumestd,
                'log_volumesum': log_volumesum,
                'event_volumecount': event_volumecount,
                'event_volumemin': event_volumemin,
                'event_volumemean': event_volumemean,
                'event_volumemax': event_volumemax,
                'event_volumestd': event_volumestd,
                'event_volumesum': event_volumesum,
                'sev_volumecount': sev_volumecount,
                'sev_volumemin': sev_volumemin,
                'sev_volumemean': sev_volumemean,
                'sev_volumemax': sev_volumemax,
                'sev_volumestd': sev_volumestd,
                'sev_volumesum': sev_volumesum,
                'log_volume': log_volume,
                'avgvol_per_loc': avgvol_per_loc,
                'maxvol_per_loc': maxvol_per_loc,
                'minvol_per_loc': minvol_per_loc,
                'medianvol_per_loc': medianvol_per_loc,
                'stdvol_per_loc': stdvol_per_loc,
                'order': order,
                'event_order': event_order,
                'feature_order': feature_order,
                'location_rank_asc': location_rank_asc,
                'location_rank_desc': location_rank_desc,
                'location_count': location_count,
                'loc_rank_rel': loc_rank_rel

                }
    data = pd.DataFrame(features, index=[0])

    return data

userinput = get_user_input()

st.subheader('User Input parameters')
st.write(userinput)


prediction_proba = classifier.predict_proba(userinput)

visualize_confidence_level(prediction_proba)

st.write("Streamlit App Created @Sriram Sripada")

st.subheader('More Information')
st.write("""
For a deeper dive into the project, please visit the [repo on GitHub](https://github.com/sriramsripadaa/telstra-end-to-end) 
where you can find all the code used in analysis, modeling, visualizations, etc.  
""")

