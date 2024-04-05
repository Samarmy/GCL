# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# All contributions by Andy Brock:
# Copyright (c) 2019 Andy Brock
#
# All contributions made by NAVER Corp.:
# Copyright (c) 2020-present NAVER Corp.
#
# MIT license

import sys
import os
import os.path

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from data_utils import utils as data_utils
from PIL import Image
import numpy as np
from tqdm import tqdm
import random
import sklearn.metrics
import torch.utils.data as data
try:
    import faiss
    USE_FAISS = 1
except:
    print('Faiss library not found!')
    USE_FAISS = 0
import h5py as h5
import torch

IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm"]

imagenet_idx = {'n01440764': 0,
 'n01443537': 1,
 'n01484850': 2,
 'n01491361': 3,
 'n01494475': 4,
 'n01496331': 5,
 'n01498041': 6,
 'n01514668': 7,
 'n01514859': 8,
 'n01518878': 9,
 'n01530575': 10,
 'n01531178': 11,
 'n01532829': 12,
 'n01534433': 13,
 'n01537544': 14,
 'n01558993': 15,
 'n01560419': 16,
 'n01580077': 17,
 'n01582220': 18,
 'n01592084': 19,
 'n01601694': 20,
 'n01608432': 21,
 'n01614925': 22,
 'n01616318': 23,
 'n01622779': 24,
 'n01629819': 25,
 'n01630670': 26,
 'n01631663': 27,
 'n01632458': 28,
 'n01632777': 29,
 'n01641577': 30,
 'n01644373': 31,
 'n01644900': 32,
 'n01664065': 33,
 'n01665541': 34,
 'n01667114': 35,
 'n01667778': 36,
 'n01669191': 37,
 'n01675722': 38,
 'n01677366': 39,
 'n01682714': 40,
 'n01685808': 41,
 'n01687978': 42,
 'n01688243': 43,
 'n01689811': 44,
 'n01692333': 45,
 'n01693334': 46,
 'n01694178': 47,
 'n01695060': 48,
 'n01697457': 49,
 'n01698640': 50,
 'n01704323': 51,
 'n01728572': 52,
 'n01728920': 53,
 'n01729322': 54,
 'n01729977': 55,
 'n01734418': 56,
 'n01735189': 57,
 'n01737021': 58,
 'n01739381': 59,
 'n01740131': 60,
 'n01742172': 61,
 'n01744401': 62,
 'n01748264': 63,
 'n01749939': 64,
 'n01751748': 65,
 'n01753488': 66,
 'n01755581': 67,
 'n01756291': 68,
 'n01768244': 69,
 'n01770081': 70,
 'n01770393': 71,
 'n01773157': 72,
 'n01773549': 73,
 'n01773797': 74,
 'n01774384': 75,
 'n01774750': 76,
 'n01775062': 77,
 'n01776313': 78,
 'n01784675': 79,
 'n01795545': 80,
 'n01796340': 81,
 'n01797886': 82,
 'n01798484': 83,
 'n01806143': 84,
 'n01806567': 85,
 'n01807496': 86,
 'n01817953': 87,
 'n01818515': 88,
 'n01819313': 89,
 'n01820546': 90,
 'n01824575': 91,
 'n01828970': 92,
 'n01829413': 93,
 'n01833805': 94,
 'n01843065': 95,
 'n01843383': 96,
 'n01847000': 97,
 'n01855032': 98,
 'n01855672': 99,
 'n01860187': 100,
 'n01871265': 101,
 'n01872401': 102,
 'n01873310': 103,
 'n01877812': 104,
 'n01882714': 105,
 'n01883070': 106,
 'n01910747': 107,
 'n01914609': 108,
 'n01917289': 109,
 'n01924916': 110,
 'n01930112': 111,
 'n01943899': 112,
 'n01944390': 113,
 'n01945685': 114,
 'n01950731': 115,
 'n01955084': 116,
 'n01968897': 117,
 'n01978287': 118,
 'n01978455': 119,
 'n01980166': 120,
 'n01981276': 121,
 'n01983481': 122,
 'n01984695': 123,
 'n01985128': 124,
 'n01986214': 125,
 'n01990800': 126,
 'n02002556': 127,
 'n02002724': 128,
 'n02006656': 129,
 'n02007558': 130,
 'n02009229': 131,
 'n02009912': 132,
 'n02011460': 133,
 'n02012849': 134,
 'n02013706': 135,
 'n02017213': 136,
 'n02018207': 137,
 'n02018795': 138,
 'n02025239': 139,
 'n02027492': 140,
 'n02028035': 141,
 'n02033041': 142,
 'n02037110': 143,
 'n02051845': 144,
 'n02056570': 145,
 'n02058221': 146,
 'n02066245': 147,
 'n02071294': 148,
 'n02074367': 149,
 'n02077923': 150,
 'n02085620': 151,
 'n02085782': 152,
 'n02085936': 153,
 'n02086079': 154,
 'n02086240': 155,
 'n02086646': 156,
 'n02086910': 157,
 'n02087046': 158,
 'n02087394': 159,
 'n02088094': 160,
 'n02088238': 161,
 'n02088364': 162,
 'n02088466': 163,
 'n02088632': 164,
 'n02089078': 165,
 'n02089867': 166,
 'n02089973': 167,
 'n02090379': 168,
 'n02090622': 169,
 'n02090721': 170,
 'n02091032': 171,
 'n02091134': 172,
 'n02091244': 173,
 'n02091467': 174,
 'n02091635': 175,
 'n02091831': 176,
 'n02092002': 177,
 'n02092339': 178,
 'n02093256': 179,
 'n02093428': 180,
 'n02093647': 181,
 'n02093754': 182,
 'n02093859': 183,
 'n02093991': 184,
 'n02094114': 185,
 'n02094258': 186,
 'n02094433': 187,
 'n02095314': 188,
 'n02095570': 189,
 'n02095889': 190,
 'n02096051': 191,
 'n02096177': 192,
 'n02096294': 193,
 'n02096437': 194,
 'n02096585': 195,
 'n02097047': 196,
 'n02097130': 197,
 'n02097209': 198,
 'n02097298': 199,
 'n02097474': 200,
 'n02097658': 201,
 'n02098105': 202,
 'n02098286': 203,
 'n02098413': 204,
 'n02099267': 205,
 'n02099429': 206,
 'n02099601': 207,
 'n02099712': 208,
 'n02099849': 209,
 'n02100236': 210,
 'n02100583': 211,
 'n02100735': 212,
 'n02100877': 213,
 'n02101006': 214,
 'n02101388': 215,
 'n02101556': 216,
 'n02102040': 217,
 'n02102177': 218,
 'n02102318': 219,
 'n02102480': 220,
 'n02102973': 221,
 'n02104029': 222,
 'n02104365': 223,
 'n02105056': 224,
 'n02105162': 225,
 'n02105251': 226,
 'n02105412': 227,
 'n02105505': 228,
 'n02105641': 229,
 'n02105855': 230,
 'n02106030': 231,
 'n02106166': 232,
 'n02106382': 233,
 'n02106550': 234,
 'n02106662': 235,
 'n02107142': 236,
 'n02107312': 237,
 'n02107574': 238,
 'n02107683': 239,
 'n02107908': 240,
 'n02108000': 241,
 'n02108089': 242,
 'n02108422': 243,
 'n02108551': 244,
 'n02108915': 245,
 'n02109047': 246,
 'n02109525': 247,
 'n02109961': 248,
 'n02110063': 249,
 'n02110185': 250,
 'n02110341': 251,
 'n02110627': 252,
 'n02110806': 253,
 'n02110958': 254,
 'n02111129': 255,
 'n02111277': 256,
 'n02111500': 257,
 'n02111889': 258,
 'n02112018': 259,
 'n02112137': 260,
 'n02112350': 261,
 'n02112706': 262,
 'n02113023': 263,
 'n02113186': 264,
 'n02113624': 265,
 'n02113712': 266,
 'n02113799': 267,
 'n02113978': 268,
 'n02114367': 269,
 'n02114548': 270,
 'n02114712': 271,
 'n02114855': 272,
 'n02115641': 273,
 'n02115913': 274,
 'n02116738': 275,
 'n02117135': 276,
 'n02119022': 277,
 'n02119789': 278,
 'n02120079': 279,
 'n02120505': 280,
 'n02123045': 281,
 'n02123159': 282,
 'n02123394': 283,
 'n02123597': 284,
 'n02124075': 285,
 'n02125311': 286,
 'n02127052': 287,
 'n02128385': 288,
 'n02128757': 289,
 'n02128925': 290,
 'n02129165': 291,
 'n02129604': 292,
 'n02130308': 293,
 'n02132136': 294,
 'n02133161': 295,
 'n02134084': 296,
 'n02134418': 297,
 'n02137549': 298,
 'n02138441': 299,
 'n02165105': 300,
 'n02165456': 301,
 'n02167151': 302,
 'n02168699': 303,
 'n02169497': 304,
 'n02172182': 305,
 'n02174001': 306,
 'n02177972': 307,
 'n02190166': 308,
 'n02206856': 309,
 'n02219486': 310,
 'n02226429': 311,
 'n02229544': 312,
 'n02231487': 313,
 'n02233338': 314,
 'n02236044': 315,
 'n02256656': 316,
 'n02259212': 317,
 'n02264363': 318,
 'n02268443': 319,
 'n02268853': 320,
 'n02276258': 321,
 'n02277742': 322,
 'n02279972': 323,
 'n02280649': 324,
 'n02281406': 325,
 'n02281787': 326,
 'n02317335': 327,
 'n02319095': 328,
 'n02321529': 329,
 'n02325366': 330,
 'n02326432': 331,
 'n02328150': 332,
 'n02342885': 333,
 'n02346627': 334,
 'n02356798': 335,
 'n02361337': 336,
 'n02363005': 337,
 'n02364673': 338,
 'n02389026': 339,
 'n02391049': 340,
 'n02395406': 341,
 'n02396427': 342,
 'n02397096': 343,
 'n02398521': 344,
 'n02403003': 345,
 'n02408429': 346,
 'n02410509': 347,
 'n02412080': 348,
 'n02415577': 349,
 'n02417914': 350,
 'n02422106': 351,
 'n02422699': 352,
 'n02423022': 353,
 'n02437312': 354,
 'n02437616': 355,
 'n02441942': 356,
 'n02442845': 357,
 'n02443114': 358,
 'n02443484': 359,
 'n02444819': 360,
 'n02445715': 361,
 'n02447366': 362,
 'n02454379': 363,
 'n02457408': 364,
 'n02480495': 365,
 'n02480855': 366,
 'n02481823': 367,
 'n02483362': 368,
 'n02483708': 369,
 'n02484975': 370,
 'n02486261': 371,
 'n02486410': 372,
 'n02487347': 373,
 'n02488291': 374,
 'n02488702': 375,
 'n02489166': 376,
 'n02490219': 377,
 'n02492035': 378,
 'n02492660': 379,
 'n02493509': 380,
 'n02493793': 381,
 'n02494079': 382,
 'n02497673': 383,
 'n02500267': 384,
 'n02504013': 385,
 'n02504458': 386,
 'n02509815': 387,
 'n02510455': 388,
 'n02514041': 389,
 'n02526121': 390,
 'n02536864': 391,
 'n02606052': 392,
 'n02607072': 393,
 'n02640242': 394,
 'n02641379': 395,
 'n02643566': 396,
 'n02655020': 397,
 'n02666196': 398,
 'n02667093': 399,
 'n02669723': 400,
 'n02672831': 401,
 'n02676566': 402,
 'n02687172': 403,
 'n02690373': 404,
 'n02692877': 405,
 'n02699494': 406,
 'n02701002': 407,
 'n02704792': 408,
 'n02708093': 409,
 'n02727426': 410,
 'n02730930': 411,
 'n02747177': 412,
 'n02749479': 413,
 'n02769748': 414,
 'n02776631': 415,
 'n02777292': 416,
 'n02782093': 417,
 'n02783161': 418,
 'n02786058': 419,
 'n02787622': 420,
 'n02788148': 421,
 'n02790996': 422,
 'n02791124': 423,
 'n02791270': 424,
 'n02793495': 425,
 'n02794156': 426,
 'n02795169': 427,
 'n02797295': 428,
 'n02799071': 429,
 'n02802426': 430,
 'n02804414': 431,
 'n02804610': 432,
 'n02807133': 433,
 'n02808304': 434,
 'n02808440': 435,
 'n02814533': 436,
 'n02814860': 437,
 'n02815834': 438,
 'n02817516': 439,
 'n02823428': 440,
 'n02823750': 441,
 'n02825657': 442,
 'n02834397': 443,
 'n02835271': 444,
 'n02837789': 445,
 'n02840245': 446,
 'n02841315': 447,
 'n02843684': 448,
 'n02859443': 449,
 'n02860847': 450,
 'n02865351': 451,
 'n02869837': 452,
 'n02870880': 453,
 'n02871525': 454,
 'n02877765': 455,
 'n02879718': 456,
 'n02883205': 457,
 'n02892201': 458,
 'n02892767': 459,
 'n02894605': 460,
 'n02895154': 461,
 'n02906734': 462,
 'n02909870': 463,
 'n02910353': 464,
 'n02916936': 465,
 'n02917067': 466,
 'n02927161': 467,
 'n02930766': 468,
 'n02939185': 469,
 'n02948072': 470,
 'n02950826': 471,
 'n02951358': 472,
 'n02951585': 473,
 'n02963159': 474,
 'n02965783': 475,
 'n02966193': 476,
 'n02966687': 477,
 'n02971356': 478,
 'n02974003': 479,
 'n02977058': 480,
 'n02978881': 481,
 'n02979186': 482,
 'n02980441': 483,
 'n02981792': 484,
 'n02988304': 485,
 'n02992211': 486,
 'n02992529': 487,
 'n02999410': 488,
 'n03000134': 489,
 'n03000247': 490,
 'n03000684': 491,
 'n03014705': 492,
 'n03016953': 493,
 'n03017168': 494,
 'n03018349': 495,
 'n03026506': 496,
 'n03028079': 497,
 'n03032252': 498,
 'n03041632': 499,
 'n03042490': 500,
 'n03045698': 501,
 'n03047690': 502,
 'n03062245': 503,
 'n03063599': 504,
 'n03063689': 505,
 'n03065424': 506,
 'n03075370': 507,
 'n03085013': 508,
 'n03089624': 509,
 'n03095699': 510,
 'n03100240': 511,
 'n03109150': 512,
 'n03110669': 513,
 'n03124043': 514,
 'n03124170': 515,
 'n03125729': 516,
 'n03126707': 517,
 'n03127747': 518,
 'n03127925': 519,
 'n03131574': 520,
 'n03133878': 521,
 'n03134739': 522,
 'n03141823': 523,
 'n03146219': 524,
 'n03160309': 525,
 'n03179701': 526,
 'n03180011': 527,
 'n03187595': 528,
 'n03188531': 529,
 'n03196217': 530,
 'n03197337': 531,
 'n03201208': 532,
 'n03207743': 533,
 'n03207941': 534,
 'n03208938': 535,
 'n03216828': 536,
 'n03218198': 537,
 'n03220513': 538,
 'n03223299': 539,
 'n03240683': 540,
 'n03249569': 541,
 'n03250847': 542,
 'n03255030': 543,
 'n03259280': 544,
 'n03271574': 545,
 'n03272010': 546,
 'n03272562': 547,
 'n03290653': 548,
 'n03291819': 549,
 'n03297495': 550,
 'n03314780': 551,
 'n03325584': 552,
 'n03337140': 553,
 'n03344393': 554,
 'n03345487': 555,
 'n03347037': 556,
 'n03355925': 557,
 'n03372029': 558,
 'n03376595': 559,
 'n03379051': 560,
 'n03384352': 561,
 'n03388043': 562,
 'n03388183': 563,
 'n03388549': 564,
 'n03393912': 565,
 'n03394916': 566,
 'n03400231': 567,
 'n03404251': 568,
 'n03417042': 569,
 'n03424325': 570,
 'n03425413': 571,
 'n03443371': 572,
 'n03444034': 573,
 'n03445777': 574,
 'n03445924': 575,
 'n03447447': 576,
 'n03447721': 577,
 'n03450230': 578,
 'n03452741': 579,
 'n03457902': 580,
 'n03459775': 581,
 'n03461385': 582,
 'n03467068': 583,
 'n03476684': 584,
 'n03476991': 585,
 'n03478589': 586,
 'n03481172': 587,
 'n03482405': 588,
 'n03483316': 589,
 'n03485407': 590,
 'n03485794': 591,
 'n03492542': 592,
 'n03494278': 593,
 'n03495258': 594,
 'n03496892': 595,
 'n03498962': 596,
 'n03527444': 597,
 'n03529860': 598,
 'n03530642': 599,
 'n03532672': 600,
 'n03534580': 601,
 'n03535780': 602,
 'n03538406': 603,
 'n03544143': 604,
 'n03584254': 605,
 'n03584829': 606,
 'n03590841': 607,
 'n03594734': 608,
 'n03594945': 609,
 'n03595614': 610,
 'n03598930': 611,
 'n03599486': 612,
 'n03602883': 613,
 'n03617480': 614,
 'n03623198': 615,
 'n03627232': 616,
 'n03630383': 617,
 'n03633091': 618,
 'n03637318': 619,
 'n03642806': 620,
 'n03649909': 621,
 'n03657121': 622,
 'n03658185': 623,
 'n03661043': 624,
 'n03662601': 625,
 'n03666591': 626,
 'n03670208': 627,
 'n03673027': 628,
 'n03676483': 629,
 'n03680355': 630,
 'n03690938': 631,
 'n03691459': 632,
 'n03692522': 633,
 'n03697007': 634,
 'n03706229': 635,
 'n03709823': 636,
 'n03710193': 637,
 'n03710637': 638,
 'n03710721': 639,
 'n03717622': 640,
 'n03720891': 641,
 'n03721384': 642,
 'n03724870': 643,
 'n03729826': 644,
 'n03733131': 645,
 'n03733281': 646,
 'n03733805': 647,
 'n03742115': 648,
 'n03743016': 649,
 'n03759954': 650,
 'n03761084': 651,
 'n03763968': 652,
 'n03764736': 653,
 'n03769881': 654,
 'n03770439': 655,
 'n03770679': 656,
 'n03773504': 657,
 'n03775071': 658,
 'n03775546': 659,
 'n03776460': 660,
 'n03777568': 661,
 'n03777754': 662,
 'n03781244': 663,
 'n03782006': 664,
 'n03785016': 665,
 'n03786901': 666,
 'n03787032': 667,
 'n03788195': 668,
 'n03788365': 669,
 'n03791053': 670,
 'n03792782': 671,
 'n03792972': 672,
 'n03793489': 673,
 'n03794056': 674,
 'n03796401': 675,
 'n03803284': 676,
 'n03804744': 677,
 'n03814639': 678,
 'n03814906': 679,
 'n03825788': 680,
 'n03832673': 681,
 'n03837869': 682,
 'n03838899': 683,
 'n03840681': 684,
 'n03841143': 685,
 'n03843555': 686,
 'n03854065': 687,
 'n03857828': 688,
 'n03866082': 689,
 'n03868242': 690,
 'n03868863': 691,
 'n03871628': 692,
 'n03873416': 693,
 'n03874293': 694,
 'n03874599': 695,
 'n03876231': 696,
 'n03877472': 697,
 'n03877845': 698,
 'n03884397': 699,
 'n03887697': 700,
 'n03888257': 701,
 'n03888605': 702,
 'n03891251': 703,
 'n03891332': 704,
 'n03895866': 705,
 'n03899768': 706,
 'n03902125': 707,
 'n03903868': 708,
 'n03908618': 709,
 'n03908714': 710,
 'n03916031': 711,
 'n03920288': 712,
 'n03924679': 713,
 'n03929660': 714,
 'n03929855': 715,
 'n03930313': 716,
 'n03930630': 717,
 'n03933933': 718,
 'n03935335': 719,
 'n03937543': 720,
 'n03938244': 721,
 'n03942813': 722,
 'n03944341': 723,
 'n03947888': 724,
 'n03950228': 725,
 'n03954731': 726,
 'n03956157': 727,
 'n03958227': 728,
 'n03961711': 729,
 'n03967562': 730,
 'n03970156': 731,
 'n03976467': 732,
 'n03976657': 733,
 'n03977966': 734,
 'n03980874': 735,
 'n03982430': 736,
 'n03983396': 737,
 'n03991062': 738,
 'n03992509': 739,
 'n03995372': 740,
 'n03998194': 741,
 'n04004767': 742,
 'n04005630': 743,
 'n04008634': 744,
 'n04009552': 745,
 'n04019541': 746,
 'n04023962': 747,
 'n04026417': 748,
 'n04033901': 749,
 'n04033995': 750,
 'n04037443': 751,
 'n04039381': 752,
 'n04040759': 753,
 'n04041544': 754,
 'n04044716': 755,
 'n04049303': 756,
 'n04065272': 757,
 'n04067472': 758,
 'n04069434': 759,
 'n04070727': 760,
 'n04074963': 761,
 'n04081281': 762,
 'n04086273': 763,
 'n04090263': 764,
 'n04099969': 765,
 'n04111531': 766,
 'n04116512': 767,
 'n04118538': 768,
 'n04118776': 769,
 'n04120489': 770,
 'n04125021': 771,
 'n04127249': 772,
 'n04131690': 773,
 'n04133789': 774,
 'n04136333': 775,
 'n04141076': 776,
 'n04141327': 777,
 'n04141975': 778,
 'n04146614': 779,
 'n04147183': 780,
 'n04149813': 781,
 'n04152593': 782,
 'n04153751': 783,
 'n04154565': 784,
 'n04162706': 785,
 'n04179913': 786,
 'n04192698': 787,
 'n04200800': 788,
 'n04201297': 789,
 'n04204238': 790,
 'n04204347': 791,
 'n04208210': 792,
 'n04209133': 793,
 'n04209239': 794,
 'n04228054': 795,
 'n04229816': 796,
 'n04235860': 797,
 'n04238763': 798,
 'n04239074': 799,
 'n04243546': 800,
 'n04251144': 801,
 'n04252077': 802,
 'n04252225': 803,
 'n04254120': 804,
 'n04254680': 805,
 'n04254777': 806,
 'n04258138': 807,
 'n04259630': 808,
 'n04263257': 809,
 'n04264628': 810,
 'n04265275': 811,
 'n04266014': 812,
 'n04270147': 813,
 'n04273569': 814,
 'n04275548': 815,
 'n04277352': 816,
 'n04285008': 817,
 'n04286575': 818,
 'n04296562': 819,
 'n04310018': 820,
 'n04311004': 821,
 'n04311174': 822,
 'n04317175': 823,
 'n04325704': 824,
 'n04326547': 825,
 'n04328186': 826,
 'n04330267': 827,
 'n04332243': 828,
 'n04335435': 829,
 'n04336792': 830,
 'n04344873': 831,
 'n04346328': 832,
 'n04347754': 833,
 'n04350905': 834,
 'n04355338': 835,
 'n04355933': 836,
 'n04356056': 837,
 'n04357314': 838,
 'n04366367': 839,
 'n04367480': 840,
 'n04370456': 841,
 'n04371430': 842,
 'n04371774': 843,
 'n04372370': 844,
 'n04376876': 845,
 'n04380533': 846,
 'n04389033': 847,
 'n04392985': 848,
 'n04398044': 849,
 'n04399382': 850,
 'n04404412': 851,
 'n04409515': 852,
 'n04417672': 853,
 'n04418357': 854,
 'n04423845': 855,
 'n04428191': 856,
 'n04429376': 857,
 'n04435653': 858,
 'n04442312': 859,
 'n04443257': 860,
 'n04447861': 861,
 'n04456115': 862,
 'n04458633': 863,
 'n04461696': 864,
 'n04462240': 865,
 'n04465501': 866,
 'n04467665': 867,
 'n04476259': 868,
 'n04479046': 869,
 'n04482393': 870,
 'n04483307': 871,
 'n04485082': 872,
 'n04486054': 873,
 'n04487081': 874,
 'n04487394': 875,
 'n04493381': 876,
 'n04501370': 877,
 'n04505470': 878,
 'n04507155': 879,
 'n04509417': 880,
 'n04515003': 881,
 'n04517823': 882,
 'n04522168': 883,
 'n04523525': 884,
 'n04525038': 885,
 'n04525305': 886,
 'n04532106': 887,
 'n04532670': 888,
 'n04536866': 889,
 'n04540053': 890,
 'n04542943': 891,
 'n04548280': 892,
 'n04548362': 893,
 'n04550184': 894,
 'n04552348': 895,
 'n04553703': 896,
 'n04554684': 897,
 'n04557648': 898,
 'n04560804': 899,
 'n04562935': 900,
 'n04579145': 901,
 'n04579432': 902,
 'n04584207': 903,
 'n04589890': 904,
 'n04590129': 905,
 'n04591157': 906,
 'n04591713': 907,
 'n04592741': 908,
 'n04596742': 909,
 'n04597913': 910,
 'n04599235': 911,
 'n04604644': 912,
 'n04606251': 913,
 'n04612504': 914,
 'n04613696': 915,
 'n06359193': 916,
 'n06596364': 917,
 'n06785654': 918,
 'n06794110': 919,
 'n06874185': 920,
 'n07248320': 921,
 'n07565083': 922,
 'n07579787': 923,
 'n07583066': 924,
 'n07584110': 925,
 'n07590611': 926,
 'n07613480': 927,
 'n07614500': 928,
 'n07615774': 929,
 'n07684084': 930,
 'n07693725': 931,
 'n07695742': 932,
 'n07697313': 933,
 'n07697537': 934,
 'n07711569': 935,
 'n07714571': 936,
 'n07714990': 937,
 'n07715103': 938,
 'n07716358': 939,
 'n07716906': 940,
 'n07717410': 941,
 'n07717556': 942,
 'n07718472': 943,
 'n07718747': 944,
 'n07720875': 945,
 'n07730033': 946,
 'n07734744': 947,
 'n07742313': 948,
 'n07745940': 949,
 'n07747607': 950,
 'n07749582': 951,
 'n07753113': 952,
 'n07753275': 953,
 'n07753592': 954,
 'n07754684': 955,
 'n07760859': 956,
 'n07768694': 957,
 'n07802026': 958,
 'n07831146': 959,
 'n07836838': 960,
 'n07860988': 961,
 'n07871810': 962,
 'n07873807': 963,
 'n07875152': 964,
 'n07880968': 965,
 'n07892512': 966,
 'n07920052': 967,
 'n07930864': 968,
 'n07932039': 969,
 'n09193705': 970,
 'n09229709': 971,
 'n09246464': 972,
 'n09256479': 973,
 'n09288635': 974,
 'n09332890': 975,
 'n09399592': 976,
 'n09421951': 977,
 'n09428293': 978,
 'n09468604': 979,
 'n09472597': 980,
 'n09835506': 981,
 'n10148035': 982,
 'n10565667': 983,
 'n11879895': 984,
 'n11939491': 985,
 'n12057211': 986,
 'n12144580': 987,
 'n12267677': 988,
 'n12620546': 989,
 'n12768682': 990,
 'n12985857': 991,
 'n12998815': 992,
 'n13037406': 993,
 'n13040303': 994,
 'n13044778': 995,
 'n13052670': 996,
 'n13054560': 997,
 'n13133613': 998,
 'n15075141': 999}


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in tqdm(sorted(os.listdir(dir))):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

      root/dogball/xxx.png
      root/dogball/xxy.png
      root/dogball/xxz.png

      root/cat/123.png
      root/cat/nsdf3.png
      root/cat/asd932_.png

  Parameters
  ----------
      root: string. Root directory path.
      transform: callable, optional. A function/transform that  takes in an PIL image
          and returns a transformed version. E.g, ``transforms.RandomCrop``
      target_transform: callable, optional. A function/transform that takes in the
          target and transforms it.
      loader: callable, optional. A function to load an image given its path.

   Attributes
   ----------
      classes: list. List of the class names.
      class_to_idx: dict. Dict with items (class_name, class_index).
      imgs: list. List of (image path, class_index) tuples
  """

    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=default_loader,
        load_in_mem=False,
        index_filename="imagenet_imgs.npz",
        longtail=False,
        subsampled=False,
        split="train",
        get_encodings=False,
        **kwargs
    ):

        classes, class_to_idx = find_classes(root)
        print(class_to_idx)
        # Load pre-computed image directory walk
        if False:  # os.path.exists(os.path.join(index_filename)):
            print("Loading pre-saved Index file %s..." % index_filename)
            imgs = np.load(os.path.join(index_filename))["imgs"]
        #   If first time, walk the folder directory and save the
        #  results to a pre-computed file.
        else:
            print("Generating  Index file %s..." % index_filename)
            if not longtail:
                imgs = make_dataset(root, class_to_idx)
                if subsampled:
                    # Same number of samples as in ImageNet-LT
                    imgs = random.sample(imgs, 115846)
            else:
                imgs = []
                print("Using long-tail version of the dataset with split ", split, "!")
                with open(
                    "BigGAN_PyTorch/imagenet_lt/ImageNet_LT_" + split + ".txt"
                ) as f:
                    for line in f:
                        imgs.append(
                            (
                                os.path.join(
                                    root, "/".join(line.split()[0].split("/")[1:])
                                ),
                                int(line.split()[1]),
                            )
                        )
            np.savez_compressed(os.path.join(index_filename), **{"imgs": imgs})
        if len(imgs) == 0:
            raise (
                RuntimeError(
                    "Found 0 images in subfolders of: " + root + "\n"
                    "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)
                )
            )

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.load_in_mem = load_in_mem
        self.get_encodings = get_encodings

        if self.load_in_mem:
            print("Loading all images into memory...")
            self.data, self.labels = [], []
            for index in tqdm(range(len(self.imgs))):
                path, target = imgs[index][0], imgs[index][1]
                self.data.append(self.transform(self.loader(path)))
                self.labels.append(target)

    def __getitem__(self, index):
        """
    Parameters
    ----------
        index: int. Index

    Returns
    -------
        tuple: (image, target) where target is class_index of the target class.
    """
        if self.load_in_mem:
            img = self.data[index]
            target = self.labels[index]
        else:
            path, target = self.imgs[index]
            img = self.loader(str(path))
            if self.transform is not None:
                img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        target = imagenet_idx[path.split("/")[-1].split("_")[0]]
        if self.get_encodings:
            encoding = np.load(path.split(".")[0] + ".npy")
            return img, int(target), index, path, encoding

        
        return img, target, index, path, 

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp, self.target_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str


class ILSVRC_HDF5_feats(data.Dataset):
    """ ILSVRC_HDF5_feats: A dataset to support I/O from an HDF5.

        Parameters
        ----------
            root :str
                Path to the hdf5 file containing images and labels.
            root_feats: str, optional
                Path to the hdf5 file containing the instance features.
            root_nns: str, optional
                Path to the hdf5 file containing the list of nearest neighbors for each instance.
            transform : callable, optional
                A function/transform that  takes in an PIL image and returns a transformed version.
                 E.g, ``transforms.RandomCrop``
            target_transform: callable, optional
                A function/transform that takes in the target and transforms it.
            load_labels: bool, optional
                Return labels for each example.
            load_features: bool, optional
                Return instance features and its neighbors (needed for IC-GAN).
            load_in_mem_images: bool, optional
                Load all images in memory.
            load_in_mem_labels: bool, optional
                Load all labels in memory.
            load_in_mem_feats: bool, optional
                Load all instance features in memory.
            k_nn: int, optional
                Size of the neighborhood obtained with the k-NN algorithm.
            which_nn_balance: str, optional
                Whether to sample an instance or a neighbor class first. By default,
                ``instance_balance`` is used.
                 Using ``nnclass_balance`` allows class balancing to be applied.
            kmeans_file: str, optional
                Path to a file where only the dataset indexes selected with k-means are stored.
                 It reduces the amount of available data to train or test the model.
            n_subsampled_data: int, optional
                If other than -1, that number of data points are randomly selected from the dataset.
                It reduces the amount of available data to train or test the model.
            filter_hd: int, optional
                Only used for COCO-Stuff dataset. If -1, all COCO-Stuff evaluation set is used.
                 If 0, only images with seen class combinations are used.
                 If 1, only images with unseen class combinations are used.
            label_dim: int, optional
                Dimensionality of label embeddings. Useful for the StyleGAN2 backbone code.
            feature_dim: int, optional
                Dimensionality of instance features embeddings. Useful for the StyleGAN2 backbone
                 code.
            feature_augmentation: bool, optional
                Use the instance features of the flipped ground-truth image instances as
                 conditioning, with a 50% probability.
            gpu_knn: bool, optional
                Accelerate k-NN faiss computation with GPUs.
            apply_norm: bool, optional
                Normalize images between [-0.5, 0.5].
            label_onehot: bool, optional
                Return labels as a one hot encoding. Useful for StyleGAN2 backbone code.

        Attributes
        ---------
            root: str
                Path to the hdf5 file containing images and labels.
            root_feats: str
                Path to the hdf5 file containing the instance features.
            root_nns: str
                Path to the hdf5 file containing the list of nearest neighbors for each
                instance.
            transform : callable
                A function/transform that  takes in an PIL image and returns a transformed version.
                E.g, ``transforms.RandomCrop``
            target_transform: callable
                A function/transform that takes in the target and transforms it.
            load_labels: bool
                Return labels for each example.
            load_features: bool
                Return instance features and its neighbors (needed for IC-GAN).
            load_in_mem_images: bool
                Load all images in memory.
            load_in_mem_labels: bool
                Load all labels in memory.
            load_in_mem_feats: bool
                Load all instance features in memory.
            feature_augmentation: bool
                Use the instance features of the flipped ground-truth image instances as conditioning,
                with a 50% probability.
            which_nn_balance: str
                Whether to sample an instance or a neighbor class first. By default,
                ``instance_balance`` is used. Using ``nnclass_balance`` allows class balancing to be
                 applied.
            apply_norm: bool
                Normalize images between [-0.5, 0.5].
            label_onehot: bool
                Return labels as a one hot encoding. Useful for StyleGAN2 backbone code.
            num_imgs: int.
                Number of data points in the dataset.
            data: NumPy array
                Image data, with the shape (num_imgs, w, h, 3), where w: width and h: height.
            labels: NumPy array
                Label data, with the shape (num_imgs, 1).
            feats: NumPy array
                Instance features data, with the shape (num_imgs, 2048).
            sample_nns: list
                List with length ``num_imgs``, that contains a list of the ``k_nn`` neighbor indexes
                 for each instance.
            sample_nn_radius: NumPy array
                Array of size (num_imgs) that stores the distance between each instance and its
                farthest(k-th) neighbor.
            possible_sampling_idxs: list
                List of all effective possible data samples. By default, it is a range(0, num_imgs).
            kmeans_samples: list
                List of indexes for samples selected with k-means algorithm.
            kth_values: NumPy array
                Distances between instances and its k-th neighbor.
        """

    def __init__(
        self,
        root,
        root_feats=None,
        root_nns=None,
        transform=None,
        target_transform=None,
        load_labels=True,
        load_features=True,
        load_in_mem_images=False,
        load_in_mem_labels=False,
        load_in_mem_feats=False,
        k_nn=4,
        which_nn_balance="instance_balance",
        kmeans_file=None,
        n_subsampled_data=-1,
        filter_hd=-1,
        label_dim=0,
        feature_dim=2048,
        feature_augmentation=False,
        gpu_knn=True,
        apply_norm=True,
        label_onehot=False,
        **kwargs
    ):
        self.root = root
        self.root_feats = root_feats
        self.root_nns = root_nns

        self.load_labels = load_labels
        self.load_features = load_features
        self._label_dim = label_dim
        self._feature_dim = feature_dim
        self.label_onehot = label_onehot
        self.feature_augmentation = feature_augmentation

        # Set the transform here
        self.transform = transform
        self.target_transform = target_transform
        # Normalization of images between -0.5 and 0.5 used in BigGAN
        self.apply_norm = apply_norm

        # load the entire dataset into memory?
        self.load_in_mem_images = load_in_mem_images
        self.load_in_mem_labels = load_in_mem_labels
        self.load_in_mem_feats = load_in_mem_feats

        self.which_nn_balance = which_nn_balance

        self.num_imgs = len(h5.File(root, "r")["labels"])

        self.labels, self.feats = None, None
        self.kth_values = None
        # If loading into memory, do so now
        print(
            "Load in mem? Images: %r, Labels: %r, Features: %r."
            % (self.load_in_mem_images, self.load_in_mem_labels, self.load_in_mem_feats)
        )
        if self.load_in_mem_images:
            print("Loading images from %s into memory..." % root)
            with h5.File(root, "r") as f:
                self.data = f["imgs"][:]
        if load_labels and self.load_in_mem_labels:
            print("Loading labels from %s into memory..." % root)
            with h5.File(root, "r") as f:
                self.labels = f["labels"][:]
        if load_features and self.load_in_mem_feats:
            print("Loading features from %s into memory..." % root_feats)
            with h5.File(root_feats, "r") as f:
                self.feats = f["feats"][:]
            # Normalizing features
            print("Normalizing features by their norm")
            self.feats /= np.linalg.norm(self.feats, axis=1, keepdims=True)
            self.feats = torch.from_numpy(self.feats)
            self.feats.share_memory_()

        if load_features:
            if root_nns is None and self.load_in_mem_feats:
                # We compute NNs only if we are loading features and there is no root_nns file.
                self._obtain_nns(k_nn, gpu=gpu_knn, faiss_lib=USE_FAISS)
            elif root_nns is not None:
                # Still loading the NNs indexes!
                print("Loading %s into memory..." % root_nns)
                with h5.File(root_nns, "r") as f:
                    self.sample_nns = f["sample_nns"][:]
                    self.sample_nn_radius = f["sample_nns_radius"][:]
            else:
                raise ValueError(
                    "If no file with pre-computed neighborhoods is provided, "
                    "the features need to be loaded in memory to extract them."
                    " Set the load_in_mem_feats=True."
                )

        # Reducing the number of available samples according to different criteria
        self.possible_sampling_idxs = range(self.num_imgs)
        self.kmeans_samples = None
        if kmeans_file is not None:
            print("Loading file  with just a few centroids (kmeans)... ", kmeans_file)
            self.kmeans_samples = np.load(kmeans_file, allow_pickle=True).item()[
                "center_examples"
            ][:, 0]
            self.possible_sampling_idxs = self.kmeans_samples
        elif n_subsampled_data > -1:
            self.possible_sampling_idxs = np.random.choice(
                np.array(self.possible_sampling_idxs),
                int(n_subsampled_data),
                replace=False,
            )
        elif filter_hd > -1:
            # For COCO_Stuff, we can divide the evaluation set in seen class combinations
            # (filter_hd=0)
            # or unseen class combinations (filter_hd=1)
            allowed_idxs = data_utils.filter_by_hd(filter_hd)
            self.possible_sampling_idxs = allowed_idxs
        # Change the size of the dataset if only a subset of the data is used
        self.possible_sampling_idxs = np.array(self.possible_sampling_idxs)
        self.num_imgs = len(self.possible_sampling_idxs)

        print(
            "All possible conditioning instances are ", len(self.possible_sampling_idxs)
        )

    def __getitem__(self, index):
        """
        Parameters
        ----------
            index: int

        Returns
        -------
            If the dataset loads both features and labels, return 4 elements: neighbor image,
             neighbor class label, instance features and instance radius
            If the dataset loads only features (no labels), return 4 elements: neighbor image,
             instance features, instance radius
            If the dataset loads ony labels (no features), return 2 elements: neighbor image and
            neighbor class label.
            If the dataset does not load features nor labels, return only an image.
        """
        # This only changes the index if possible_sampling_idx contains only a subset of the data
        # (k-means/random sampling or evaluation sets in COCO-Stuff)
        index = self.possible_sampling_idxs[index]
        img = self._get_image(index)
        target = self.get_label(index)
        if self.load_features:
            img_nn, label_nn, feats, radii = self._get_instance_features_and_nn(index)
            img = img_nn
            target = label_nn
        else:
            feats, radii = None, None

        # Apply transform
        img = torch.from_numpy(img)
        if self.apply_norm:
            img = ((img.float() / 255) - 0.5) * 2
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.label_onehot:
            target = int(target)

        if self.load_features and self.load_labels:
            return img, target, feats, radii
        elif self.load_features:
            return img, feats, radii
        elif self.load_labels:
            return img, target
        else:
            return img

    def sample_conditioning_instance_balance(self, batch_size, weights=None):
        """
        It samples a batch size of conditionings.

        First, by first sampling an instance, and then one of the neighbor's class.

        Parameters
        ----------
        batch_size: int
            Number of conditioning to sample.
        weights: NumPy array, optional
            Array of size len(self.possible_sampling_idxs), indicating the weight for each instance,
             used for sampling.

        Returns
        -------
        labels_gen: torch.LongTensor
            Tensor of shape (batch_size, label_dim). Batch of neighbor labels.
        instance_gen: torch.LongTensor
            Tensor of shape (batch_size, label_dim). Batch of instance features.
        """
        # Control instance (center of k-NN) balancing with weights
        # Sampling from p(h)
        if weights is None:
            # np.random.randint is a faster function than np.random.choice.
            # If there is no sampling weights, use this one.
            sel_idxs = np.random.randint(0, len(self.possible_sampling_idxs), size=batch_size)
            sel_idxs = self.possible_sampling_idxs[sel_idxs]
        else:
            sel_idxs = np.random.choice(
                self.possible_sampling_idxs, batch_size, replace=True, p=weights
            )

        # Features from center example
        instance_gen = self.get_instance_features(sel_idxs)
        # Get labels from neighbor
        labels_gen = []
        for idx_ in sel_idxs:
            # Sampling neighbor from p(x_nn, y_nn| h)
            chosen_idx = np.random.choice(self.sample_nns[idx_])
            # Labels from neighbors
            if self.load_labels:
                labels_gen.append(self.get_label(chosen_idx)[np.newaxis, ...])
        if self.load_labels:
            labels_gen = np.concatenate(labels_gen, 0)
            labels_gen = torch.LongTensor(labels_gen)
        else:
            labels_gen = None

        instance_gen = torch.FloatTensor(instance_gen)

        return labels_gen, instance_gen

    def sample_conditioning_nnclass_balance(
        self, batch_size, weights=None, num_classes=1000
    ):
        """
        It samples a batch size of conditionings.

        First, by sampling a class, then an image from this class, and finally an instance feature
        that would have this image as a neighbor in feature space.

        Parameters
        ----------
        batch_size: int
            Number of conditioning to sample.
        weights: NumPy array, optional
            Array of size num_classes, indicating the weight for each instance, used for sampling.
        num_classes: int, optional
            Number of classes in the dataset

        Returns
        -------
        labels_gen: torch.LongTensor
            Tensor of shape (batch_size, label_dim). Batch of neighbor labels.
        instance_gen: torch.LongTensor
            Tensor of shape (batch_size, label_dim). Batch of instance features.
        """
        if weights is not None:
            weights = np.array(weights) / sum(weights)

        # Sampling from p(y)
        chosen_class = np.random.choice(
            range(num_classes), batch_size, replace=True, p=weights
        )
        nn_idxs = []
        for lab_ in chosen_class:
            # Sampling from p(x_nn|y)
            chosen_xnn = np.random.choice((self.labels == lab_).nonzero()[0])
            # Sampling from p(h| x_nn,y)
            nn_idxs.append(np.random.choice(self.sample_nns[chosen_xnn]))

        instance_gen = self.get_instance_features(nn_idxs)

        instance_gen = torch.FloatTensor(instance_gen)
        labels_gen = torch.LongTensor(chosen_class)

        return labels_gen, instance_gen

    def get_label(self, index):
        """Obtain a label as an int or as a one-hot vector."""
        if not self.load_labels:
            if self.label_onehot:
                return np.zeros(self.label_dim, dtype=np.float32).copy()
            else:
                return 0

        if self.load_labels:
            if self.load_in_mem_labels:
                target = self.labels[index]
            else:
                with h5.File(self.root, "r") as f:
                    target = f["labels"][index]
        else:
            target = None
        if self.label_onehot:
            onehot_vec = np.zeros(self.label_dim, dtype=np.float32)
            onehot_vec[target] = 1
            target = onehot_vec.copy()

        return target

    def get_instance_features(self, index):
        """Obtain an instance feature vector."""
        if not self.load_features:
            return np.zeros(self.feature_dim, dtype=np.float32).copy()

        if self.load_in_mem_feats:
            feat = self.feats[index].clone().float()  # .astype('float')
        else:
            with h5.File(self.root_feats, "r") as f:
                if isinstance(index, (int, np.int64)):
                    hflip = np.random.randint(2) == 1
                    if self.feature_augmentation and hflip:
                        feat = f["feats_hflip"][index].astype("float")
                    else:
                        feat = f["feats"][index].astype("float")
                    feat /= np.linalg.norm(feat, keepdims=True)
                else:
                    feat = []
                    for sl_idx in index:
                        hflip = np.random.randint(2) == 1
                        if self.feature_augmentation and hflip:
                            feat.append(
                                f["feats_hflip"][sl_idx].astype("float")[
                                    np.newaxis, ...
                                ]
                            )
                        else:
                            feat.append(
                                f["feats"][sl_idx].astype("float")[np.newaxis, ...]
                            )
                    feat = np.concatenate(feat)
                    feat /= np.linalg.norm(feat, axis=1, keepdims=True)
        return feat

    @property
    def resolution(self):
        with h5.File(self.root, "r") as f:
            sze = list(f["imgs"][0].shape)
        return sze[1]

    @property
    def label_dim(self):
        return self._label_dim

    @property
    def feature_dim(self):
        return self._feature_dim

    def _obtain_nns(self, k_nn=20, faiss_lib=True, feat_sz=2048, gpu=True):
        """
        It obtains the neighborhoods for all instances using the k-NN algorithm.

        Parameters
        ----------
        k_nn: int, optional
            Number of neighbors (k).
        faiss_lib: bool, optional
           If True, use the faiss library implementation of k-NN. If not, use the slower
            implementation of sklearn.
        feat_sz: int, optional
            Feature dimensionality.
        gpu: bool, optional
            If True, leverage GPU resources to speed up computation with the faiss library.

        """
        # K_nn computation takes into account the input sample as the first NN,
        # so we add an extra NN to later remove the input sample.
        k_nn += 1

        print("HERE")
        print("HERE")
        print("HERE")
        print("self.num_imgs", self.num_imgs)
        print("HERE")
        print("HERE")
        print("HERE")
        self.sample_nns = [[] for _ in range(self.num_imgs)]
        self.sample_nn_radius = np.zeros(self.num_imgs, dtype=float)

        if faiss_lib:
            cpu_index = faiss.IndexFlatL2(feat_sz)
            if gpu:
                gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)  # build the index
                index = gpu_index
            else:
                index = cpu_index
            index.add(self.feats.float().numpy().astype("float32"))
            kth_values, kth_values_arg = index.search(
                self.feats.numpy().astype("float32"), k_nn
            )
            self.kth_values = np.sqrt(kth_values)
            knn_radii = np.sqrt(kth_values[:, -1])

        else:
            dists = sklearn.metrics.pairwise_distances(
                self.feats, self.feats, metric="euclidean", n_jobs=-1
            )
            print("Computed distances.")
            knn_radii, kth_values_arg = self._get_kth_value_accurate(dists, k_nn)
        for i_sample in range(self.num_imgs):
            knns = kth_values_arg[i_sample]
            # Discarding the input sample, also seen as the 0-NN.
            knns = np.delete(knns, np.where(knns == i_sample)[0], 0)
            print("THISONE")
            print("knns.tolist()", knns.tolist())
            self.sample_nns[i_sample] = knns.tolist()
            self.sample_nn_radius[i_sample] = knn_radii[i_sample]
        print("Computed NNs.")

    @staticmethod
    def _get_kth_value_accurate(distances, k, axis=-1):
        """ Find k nearest neighbor
        Parameters
        ---------
        distances: NumPy array
            Matrix of size (M, M) of unordered distances.
        k: int
            Neighborhood size
        axis: int

        Returns
        -------
        kth values: NumPy array
            Distances of the k-th nearest neighbor along the designated axis.
        indices: NumPy array
            Array positions in the input matrix indicating all neighbors up until the k-th.

        """
        indices = np.argpartition(distances, k - 1, axis=axis)[..., :k]
        k_smallests = np.take_along_axis(distances, indices, axis=axis)
        kth_values = k_smallests.max(axis=axis)
        return kth_values, indices

    def _get_image(self, index):
        """Obtain an image array."""
        if self.load_in_mem_images:
            img = self.data[index]
        else:
            with h5.File(self.root, "r") as f:
                img = f["imgs"][index]
        return img

    def _get_instance_features_and_nn(self, index):
        """ Builds a quadruplet of neighbor image, its label, conditioning instance features, radii.

        Returns
        ----------
        img_nn: NumPy array
            Neighbor image.
        label_nn: NumPy array
            Neighbor label.
        feats: NumPy array
            Conditioning instance features.
        radii: float
            Distance between conditioning instance and farthest (k-th) neighbor.
        """
        # Standard sampling: Obtain a feature vector for the input index,
        # and image/class label for a neighbor.
        if self.which_nn_balance == "instance_balance":
            idx_h = index
            # If we are only using a selected number of instances (kmeans), re-choose the index
            if self.kmeans_samples is not None:
                index = np.random.choice(self.kmeans_samples)
            idx_nn = np.random.choice(self.sample_nns[index])

        # Reverse sampling, used when we want to perform class balancing (long-tail setup).
        # In class-conditional IC-GAN, the classes are taken from the neighbors.
        # The reverse sampling allows us to control the class balancing by using extra weights
        # in the DataLoader.
        elif self.which_nn_balance == "nnclass_balance":
            idx_h = np.random.choice(self.sample_nns[index])
            idx_nn = index

        # Index selects the instance feature vector
        radii = self.sample_nn_radius[idx_h]

        img_nn = self._get_image(idx_nn)
        label_nn = self.get_label(idx_nn)
        feats = self.get_instance_features(idx_h)

        return img_nn, label_nn, feats, radii

    def __len__(self):
        return self.num_imgs
