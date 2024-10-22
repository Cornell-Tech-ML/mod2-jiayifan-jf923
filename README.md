[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/YFgwt0yY)
# MiniTorch Module 2

<img src="https://minitorch.github.io/minitorch.svg" width="50%">


* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module2/module2/

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py project/run_manual.py project/run_scalar.py project/datasets.py



## Task 2.5: Training

### 1. Simple Dataset
Dataset config:
```python
PTS = 60
DATASET = minitorch.datasets["Simple"](PTS)
HIDDEN = 4
RATE = 0.05
```
Number of epochs: 1000

Final Image:

<img src="https://github.com/user-attachments/assets/830935a7-f7a7-4a89-996f-9b8ea21e7f84" width="75%">

Loss Graph:

<img src="https://github.com/user-attachments/assets/8ef3be86-bb4e-4631-b244-60a9dddb6184" width="75%">

Time per epoch: 0.057s

Loss log:

Epoch: 0/1000, loss: 0, correct: 0
Epoch: 10/1000, loss: 43.31654180402006, correct: 28
Epoch: 20/1000, loss: 38.22374629031152, correct: 50
Epoch: 30/1000, loss: 37.24819189375717, correct: 45
Epoch: 40/1000, loss: 36.61148516497229, correct: 42
Epoch: 50/1000, loss: 35.9956519050932, correct: 42
Epoch: 60/1000, loss: 35.33960560252612, correct: 46
Epoch: 70/1000, loss: 34.65800813117939, correct: 46
Epoch: 80/1000, loss: 33.93894692010327, correct: 47
Epoch: 90/1000, loss: 33.170331451265646, correct: 47
Epoch: 100/1000, loss: 32.34192268197617, correct: 49
Epoch: 110/1000, loss: 31.44759499168929, correct: 48
Epoch: 120/1000, loss: 30.48864953506112, correct: 48
Epoch: 130/1000, loss: 29.471699645024707, correct: 49
Epoch: 140/1000, loss: 28.438031409591083, correct: 51
Epoch: 150/1000, loss: 27.355613611721996, correct: 51
Epoch: 160/1000, loss: 26.289183341027552, correct: 51
Epoch: 170/1000, loss: 25.216518048863218, correct: 52
Epoch: 180/1000, loss: 24.06540871998321, correct: 54
Epoch: 190/1000, loss: 23.013632956869973, correct: 54
Epoch: 200/1000, loss: 22.015954612516456, correct: 54
Epoch: 210/1000, loss: 21.068147698835784, correct: 54
Epoch: 220/1000, loss: 20.177358715990824, correct: 54
Epoch: 230/1000, loss: 19.336946655347994, correct: 54
Epoch: 240/1000, loss: 18.54611137299473, correct: 54
Epoch: 250/1000, loss: 17.79006107287512, correct: 55
Epoch: 260/1000, loss: 17.073933766628596, correct: 55
Epoch: 270/1000, loss: 16.389944910296173, correct: 55
Epoch: 280/1000, loss: 15.763723142487702, correct: 55
Epoch: 290/1000, loss: 15.16622082280307, correct: 55
Epoch: 300/1000, loss: 14.596206045070792, correct: 55
Epoch: 310/1000, loss: 14.052670319304728, correct: 55
Epoch: 320/1000, loss: 13.533014532647956, correct: 57
Epoch: 330/1000, loss: 13.035732845121602, correct: 58
Epoch: 340/1000, loss: 12.560534853340448, correct: 58
Epoch: 350/1000, loss: 12.10927209094122, correct: 58
Epoch: 360/1000, loss: 11.680786792470577, correct: 59
Epoch: 370/1000, loss: 11.27228713793578, correct: 59
Epoch: 380/1000, loss: 10.883045934221585, correct: 59
Epoch: 390/1000, loss: 10.512286482196012, correct: 59
Epoch: 400/1000, loss: 10.15941506276769, correct: 59
Epoch: 410/1000, loss: 9.826515057188823, correct: 59
Epoch: 420/1000, loss: 9.509471991340797, correct: 59
Epoch: 430/1000, loss: 9.207345438563276, correct: 60
Epoch: 440/1000, loss: 8.91935148161352, correct: 60
Epoch: 450/1000, loss: 8.64473439229781, correct: 60
Epoch: 460/1000, loss: 8.38283833631408, correct: 60
Epoch: 470/1000, loss: 8.13319992001666, correct: 60
Epoch: 480/1000, loss: 7.8949429434924605, correct: 60
Epoch: 490/1000, loss: 7.667565244212927, correct: 60
Epoch: 500/1000, loss: 7.451139362398351, correct: 60
Epoch: 510/1000, loss: 7.247093059719187, correct: 60
Epoch: 520/1000, loss: 7.051979099312271, correct: 60
Epoch: 530/1000, loss: 6.865046406108906, correct: 60
Epoch: 540/1000, loss: 6.6865651862812445, correct: 60
Epoch: 550/1000, loss: 6.515412219965502, correct: 60
Epoch: 560/1000, loss: 6.351177728269361, correct: 60
Epoch: 570/1000, loss: 6.19342361836913, correct: 60
Epoch: 580/1000, loss: 6.04211829363999, correct: 60
Epoch: 590/1000, loss: 5.8973515843296545, correct: 60
Epoch: 600/1000, loss: 5.758204840265915, correct: 60
Epoch: 610/1000, loss: 5.624314927842447, correct: 60
Epoch: 620/1000, loss: 5.495620784927474, correct: 60
Epoch: 630/1000, loss: 5.370824609583299, correct: 60
Epoch: 640/1000, loss: 5.251036028189593, correct: 60
Epoch: 650/1000, loss: 5.136173805160876, correct: 60
Epoch: 660/1000, loss: 5.026523432771388, correct: 60
Epoch: 670/1000, loss: 4.920723311837841, correct: 60
Epoch: 680/1000, loss: 4.818551080160345, correct: 60
Epoch: 690/1000, loss: 4.719827387647262, correct: 60
Epoch: 700/1000, loss: 4.624390302661736, correct: 60
Epoch: 710/1000, loss: 4.532072246272434, correct: 60
Epoch: 720/1000, loss: 4.442780692760959, correct: 60
Epoch: 730/1000, loss: 4.356359992507779, correct: 60
Epoch: 740/1000, loss: 4.272688941993219, correct: 60
Epoch: 750/1000, loss: 4.1916519114233255, correct: 60
Epoch: 760/1000, loss: 4.113199265483564, correct: 60
Epoch: 770/1000, loss: 4.0373328314664345, correct: 60
Epoch: 780/1000, loss: 3.963783334461714, correct: 60
Epoch: 790/1000, loss: 3.8924510010587094, correct: 60
Epoch: 800/1000, loss: 3.823261481446333, correct: 60
Epoch: 810/1000, loss: 3.756122659192284, correct: 60
Epoch: 820/1000, loss: 3.690956958710679, correct: 60
Epoch: 830/1000, loss: 3.6276911099458484, correct: 60
Epoch: 840/1000, loss: 3.5662888871469525, correct: 60
Epoch: 850/1000, loss: 3.506616194458101, correct: 60
Epoch: 860/1000, loss: 3.4487584881191804, correct: 60
Epoch: 870/1000, loss: 3.393030577736078, correct: 60
Epoch: 880/1000, loss: 3.338851100869642, correct: 60
Epoch: 890/1000, loss: 3.286252626515355, correct: 60
Epoch: 900/1000, loss: 3.2351449869713846, correct: 60
Epoch: 910/1000, loss: 3.185422413958992, correct: 60
Epoch: 920/1000, loss: 3.137014024907704, correct: 60
Epoch: 930/1000, loss: 3.0898990345475976, correct: 60
Epoch: 940/1000, loss: 3.0440210810074855, correct: 60
Epoch: 950/1000, loss: 2.999345633716452, correct: 60
Epoch: 960/1000, loss: 2.9558275848611455, correct: 60
Epoch: 970/1000, loss: 2.9134350955819888, correct: 60
Epoch: 980/1000, loss: 2.8721225919648186, correct: 60
Epoch: 990/1000, loss: 2.8318638721730416, correct: 60
Epoch: 1000/1000, loss: 2.792617548088359, correct: 60



### 2. Diagonal Dataset
Dataset config:
```python
PTS = 60
DATASET = minitorch.datasets["Diag"](PTS)
HIDDEN = 5
RATE = 0.05
```
Number of epochs: 1000

Time per epoch: 0.075s

Final Image:

<img src="https://github.com/user-attachments/assets/a749f733-3db8-46a2-85a2-8359881d4e86" width="75%">

Loss Graph:

<img src="https://github.com/user-attachments/assets/6e1bad04-c8ba-4e82-926e-0d336a03f8af" width="75%">

Loss log:

Epoch: 0/1000, loss: 0, correct: 0
Epoch: 10/1000, loss: 35.013280854697875, correct: 48
Epoch: 20/1000, loss: 32.07496308552205, correct: 48
Epoch: 30/1000, loss: 30.55677271346856, correct: 48
Epoch: 40/1000, loss: 29.742918683477306, correct: 48
Epoch: 50/1000, loss: 29.302039101229635, correct: 48
Epoch: 60/1000, loss: 29.056663177248847, correct: 48
Epoch: 70/1000, loss: 28.90538365621854, correct: 48
Epoch: 80/1000, loss: 28.797692066614037, correct: 48
Epoch: 90/1000, loss: 28.7088579276798, correct: 48
Epoch: 100/1000, loss: 28.627160407143016, correct: 48
Epoch: 110/1000, loss: 28.54714153947819, correct: 48
Epoch: 120/1000, loss: 28.46804517843669, correct: 48
Epoch: 130/1000, loss: 28.387911203392466, correct: 48
Epoch: 140/1000, loss: 28.305187792096714, correct: 48
Epoch: 150/1000, loss: 28.21975105146169, correct: 48
Epoch: 160/1000, loss: 28.13261326106203, correct: 48
Epoch: 170/1000, loss: 28.0420212229112, correct: 48
Epoch: 180/1000, loss: 27.94772978747421, correct: 48
Epoch: 190/1000, loss: 27.84949112371968, correct: 48
Epoch: 200/1000, loss: 27.74704595423202, correct: 48
Epoch: 210/1000, loss: 27.64011863343844, correct: 48
Epoch: 220/1000, loss: 27.528413724883045, correct: 48
Epoch: 230/1000, loss: 27.411613064133427, correct: 48
Epoch: 240/1000, loss: 27.2893728589242, correct: 48
Epoch: 250/1000, loss: 27.161320615245746, correct: 48
Epoch: 260/1000, loss: 27.027051776377736, correct: 48
Epoch: 270/1000, loss: 26.886126002204943, correct: 48
Epoch: 280/1000, loss: 26.738063033006533, correct: 48
Epoch: 290/1000, loss: 26.58233809022338, correct: 48
Epoch: 300/1000, loss: 26.41837677345733, correct: 48
Epoch: 310/1000, loss: 26.245549422126242, correct: 48
Epoch: 320/1000, loss: 26.063222206670254, correct: 48
Epoch: 330/1000, loss: 25.870704782692552, correct: 48
Epoch: 340/1000, loss: 25.667043241077142, correct: 48
Epoch: 350/1000, loss: 25.451321304485944, correct: 48
Epoch: 360/1000, loss: 25.222529244695384, correct: 48
Epoch: 370/1000, loss: 24.979558697569928, correct: 48
Epoch: 380/1000, loss: 24.721203872009987, correct: 48
Epoch: 390/1000, loss: 24.44612137568589, correct: 48
Epoch: 400/1000, loss: 24.152861989085842, correct: 48
Epoch: 410/1000, loss: 23.839886668819663, correct: 48
Epoch: 420/1000, loss: 23.50551729610816, correct: 48
Epoch: 430/1000, loss: 23.147985326289074, correct: 48
Epoch: 440/1000, loss: 22.765410435120682, correct: 48
Epoch: 450/1000, loss: 22.35598416344904, correct: 48
Epoch: 460/1000, loss: 21.917827738272848, correct: 48
Epoch: 470/1000, loss: 21.44933836239425, correct: 48
Epoch: 480/1000, loss: 20.948936663204005, correct: 48
Epoch: 490/1000, loss: 20.4093385382342, correct: 48
Epoch: 500/1000, loss: 19.830690455556606, correct: 48
Epoch: 510/1000, loss: 19.190186170500535, correct: 48
Epoch: 520/1000, loss: 18.508314412427907, correct: 48
Epoch: 530/1000, loss: 17.740063473555203, correct: 48
Epoch: 540/1000, loss: 16.96277585046076, correct: 49
Epoch: 550/1000, loss: 16.208431654008855, correct: 49
Epoch: 560/1000, loss: 15.45836668170404, correct: 49
Epoch: 570/1000, loss: 14.76652366196817, correct: 53
Epoch: 580/1000, loss: 14.113236133856766, correct: 53
Epoch: 590/1000, loss: 13.476142815227144, correct: 54
Epoch: 600/1000, loss: 12.866262761866356, correct: 54
Epoch: 610/1000, loss: 12.349939499496882, correct: 54
Epoch: 620/1000, loss: 11.891210372466336, correct: 54
Epoch: 630/1000, loss: 11.470772438270961, correct: 54
Epoch: 640/1000, loss: 11.077022928459318, correct: 54
Epoch: 650/1000, loss: 10.695944915522803, correct: 55
Epoch: 660/1000, loss: 10.328116196251823, correct: 57
Epoch: 670/1000, loss: 9.972080444992901, correct: 58
Epoch: 680/1000, loss: 9.628254356733924, correct: 59
Epoch: 690/1000, loss: 9.300453149788979, correct: 59
Epoch: 700/1000, loss: 8.985002321532583, correct: 59
Epoch: 710/1000, loss: 8.681604530007041, correct: 59
Epoch: 720/1000, loss: 8.395229068801413, correct: 59
Epoch: 730/1000, loss: 8.123078124501971, correct: 59
Epoch: 740/1000, loss: 7.8629282671102, correct: 59
Epoch: 750/1000, loss: 7.612686984847594, correct: 59
Epoch: 760/1000, loss: 7.371747233726694, correct: 60
Epoch: 770/1000, loss: 7.140293728111399, correct: 60
Epoch: 780/1000, loss: 6.917771364693583, correct: 60
Epoch: 790/1000, loss: 6.7040698325066375, correct: 60
Epoch: 800/1000, loss: 6.499048622235215, correct: 60
Epoch: 810/1000, loss: 6.302449206636935, correct: 60
Epoch: 820/1000, loss: 6.113999061771453, correct: 60
Epoch: 830/1000, loss: 5.933465596099248, correct: 60
Epoch: 840/1000, loss: 5.760586281660337, correct: 60
Epoch: 850/1000, loss: 5.595159058362563, correct: 60
Epoch: 860/1000, loss: 5.436913642609816, correct: 60
Epoch: 870/1000, loss: 5.285444213685623, correct: 60
Epoch: 880/1000, loss: 5.140615173095369, correct: 60
Epoch: 890/1000, loss: 5.0020896444053635, correct: 60
Epoch: 900/1000, loss: 4.869526490023798, correct: 60
Epoch: 910/1000, loss: 4.742716292600047, correct: 60
Epoch: 920/1000, loss: 4.621436254635722, correct: 60
Epoch: 930/1000, loss: 4.505360936424169, correct: 60
Epoch: 940/1000, loss: 4.394236261683696, correct: 60
Epoch: 950/1000, loss: 4.287807585163268, correct: 60
Epoch: 960/1000, loss: 4.185844099221921, correct: 60
Epoch: 970/1000, loss: 4.088127289104387, correct: 60
Epoch: 980/1000, loss: 3.994438760291046, correct: 60
Epoch: 990/1000, loss: 3.904569251281975, correct: 60
Epoch: 1000/1000, loss: 3.8183197097876436, correct: 60


### 3. Split Dataset
Dataset config:
```python
PTS = 60
DATASET = minitorch.datasets["Xor"](PTS)
HIDDEN = 10
RATE = 0.05
```
Number of epochs: 1000

Final Image:

<img src="https://github.com/user-attachments/assets/a26aa42c-1b7f-4110-9c8f-1fe3e3df64d7" width="75%">

Loss Graph:

<img src="https://github.com/user-attachments/assets/8902f81b-fdaf-4f44-89b3-146b38844479" width="75%">

Time per epoch: 0.203s

Loss log:

Epoch: 0/1000, loss: 0, correct: 0
Epoch: 10/1000, loss: 37.12374720575592, correct: 41
Epoch: 20/1000, loss: 36.77415428354038, correct: 41
Epoch: 30/1000, loss: 36.60951308250992, correct: 41
Epoch: 40/1000, loss: 36.469501391550004, correct: 41
Epoch: 50/1000, loss: 36.33017685315294, correct: 41
Epoch: 60/1000, loss: 36.20652348889174, correct: 41
Epoch: 70/1000, loss: 36.0831992305601, correct: 41
Epoch: 80/1000, loss: 35.96896908430402, correct: 41
Epoch: 90/1000, loss: 35.861685476144885, correct: 41
Epoch: 100/1000, loss: 35.75317439566309, correct: 41
Epoch: 110/1000, loss: 35.61044303531124, correct: 41
Epoch: 120/1000, loss: 35.2725048646104, correct: 41
Epoch: 130/1000, loss: 34.87479200185997, correct: 41
Epoch: 140/1000, loss: 34.61398213573634, correct: 41
Epoch: 150/1000, loss: 34.41906030476309, correct: 41
Epoch: 160/1000, loss: 34.25375784800665, correct: 41
Epoch: 170/1000, loss: 34.08176660576898, correct: 41
Epoch: 180/1000, loss: 33.898494817332384, correct: 42
Epoch: 190/1000, loss: 33.71883127246652, correct: 42
Epoch: 200/1000, loss: 33.555953194102635, correct: 42
Epoch: 210/1000, loss: 33.40116221915323, correct: 42
Epoch: 220/1000, loss: 33.25563618830541, correct: 43
Epoch: 230/1000, loss: 33.0993924278059, correct: 43
Epoch: 240/1000, loss: 32.895740712284514, correct: 45
Epoch: 250/1000, loss: 32.70622205554645, correct: 46
Epoch: 260/1000, loss: 32.447620257176844, correct: 47
Epoch: 270/1000, loss: 31.98009389363748, correct: 48
Epoch: 280/1000, loss: 31.55641452758517, correct: 50
Epoch: 290/1000, loss: 31.25832850563258, correct: 50
Epoch: 300/1000, loss: 30.99718355199528, correct: 50
Epoch: 310/1000, loss: 30.79217061380025, correct: 50
Epoch: 320/1000, loss: 30.546217082622377, correct: 50
Epoch: 330/1000, loss: 30.05493394724949, correct: 50
Epoch: 340/1000, loss: 29.445786993903745, correct: 50
Epoch: 350/1000, loss: 29.078751092023417, correct: 50
Epoch: 360/1000, loss: 28.752934255677847, correct: 50
Epoch: 370/1000, loss: 28.438604977739836, correct: 50
Epoch: 380/1000, loss: 28.146329318883247, correct: 50
Epoch: 390/1000, loss: 27.85986621587236, correct: 50
Epoch: 400/1000, loss: 27.60277851463662, correct: 50
Epoch: 410/1000, loss: 27.355073644655636, correct: 50
Epoch: 420/1000, loss: 27.10614454158089, correct: 50
Epoch: 430/1000, loss: 26.849409815681504, correct: 50
Epoch: 440/1000, loss: 26.58600889511712, correct: 50
Epoch: 450/1000, loss: 26.32510497574185, correct: 50
Epoch: 460/1000, loss: 26.0644232212484, correct: 50
Epoch: 470/1000, loss: 25.80088441151982, correct: 50
Epoch: 480/1000, loss: 25.53401459331615, correct: 50
Epoch: 490/1000, loss: 25.263493918781588, correct: 50
Epoch: 500/1000, loss: 24.989364014126156, correct: 50
Epoch: 510/1000, loss: 24.71384123571718, correct: 50
Epoch: 520/1000, loss: 24.43787864421346, correct: 50
Epoch: 530/1000, loss: 24.159666078089575, correct: 50
Epoch: 540/1000, loss: 23.879889260794066, correct: 50
Epoch: 550/1000, loss: 23.600302353584905, correct: 50
Epoch: 560/1000, loss: 23.31816278436276, correct: 50
Epoch: 570/1000, loss: 23.03337799387935, correct: 50
Epoch: 580/1000, loss: 22.746728351687924, correct: 51
Epoch: 590/1000, loss: 22.458485112284183, correct: 51
Epoch: 600/1000, loss: 22.16914870740493, correct: 51
Epoch: 610/1000, loss: 21.878421777186393, correct: 51
Epoch: 620/1000, loss: 21.586105078538758, correct: 51
Epoch: 630/1000, loss: 21.292851051212452, correct: 51
Epoch: 640/1000, loss: 20.994055143985484, correct: 51
Epoch: 650/1000, loss: 20.69332775141811, correct: 51
Epoch: 660/1000, loss: 20.391525797038508, correct: 52
Epoch: 670/1000, loss: 20.089284540014038, correct: 53
Epoch: 680/1000, loss: 19.786910298161644, correct: 53
Epoch: 690/1000, loss: 19.484309548346562, correct: 53
Epoch: 700/1000, loss: 19.181680790311077, correct: 53
Epoch: 710/1000, loss: 18.880545968925034, correct: 53
Epoch: 720/1000, loss: 18.5804313778762, correct: 54
Epoch: 730/1000, loss: 18.28150966415054, correct: 54
Epoch: 740/1000, loss: 17.98370501239071, correct: 54
Epoch: 750/1000, loss: 17.68720329680974, correct: 54
Epoch: 760/1000, loss: 17.392358498266645, correct: 54
Epoch: 770/1000, loss: 17.099405171421896, correct: 54
Epoch: 780/1000, loss: 16.808820971681445, correct: 54
Epoch: 790/1000, loss: 16.52202973716716, correct: 54
Epoch: 800/1000, loss: 16.23868434176545, correct: 54
Epoch: 810/1000, loss: 15.959003328254175, correct: 54
Epoch: 820/1000, loss: 15.682872342253846, correct: 55
Epoch: 830/1000, loss: 15.413409275457767, correct: 55
Epoch: 840/1000, loss: 15.150619515479361, correct: 55
Epoch: 850/1000, loss: 14.892534626354315, correct: 56
Epoch: 860/1000, loss: 14.63896205510204, correct: 58
Epoch: 870/1000, loss: 14.39028509622984, correct: 58
Epoch: 880/1000, loss: 14.146946311356515, correct: 58
Epoch: 890/1000, loss: 13.908979324087714, correct: 58
Epoch: 900/1000, loss: 13.675752587176579, correct: 59
Epoch: 910/1000, loss: 13.448431321513706, correct: 59
Epoch: 920/1000, loss: 13.228503381238395, correct: 59
Epoch: 930/1000, loss: 13.014538994887848, correct: 59
Epoch: 940/1000, loss: 12.805098875989781, correct: 59
Epoch: 950/1000, loss: 12.599252583992719, correct: 59
Epoch: 960/1000, loss: 12.39704837648359, correct: 59
Epoch: 970/1000, loss: 12.198533494633967, correct: 59
Epoch: 980/1000, loss: 12.004011836449838, correct: 59
Epoch: 990/1000, loss: 11.812563260705668, correct: 59
Epoch: 1000/1000, loss: 11.624523882004114, correct: 59


### 4. Xor Dataset
Dataset config:
```python
PTS = 60
DATASET = minitorch.datasets["Xor"](PTS)
HIDDEN = 10
RATE = 0.05
```
Number of epochs: 1000

Final Image:

<img src="https://github.com/user-attachments/assets/60ab1dd2-d4a1-49b5-b645-bb9c827b9191" width="75%">

Loss Graph:

<img src="https://github.com/user-attachments/assets/f64f85d5-eb9a-4b8d-8156-10db0d867cc5" width="75%">

Time per epoch: 0.204s.

Loss log:
Epoch: 0/1500, loss: 0, correct: 0
Epoch: 10/1500, loss: 38.638894321966376, correct: 40
Epoch: 20/1500, loss: 37.54326542046629, correct: 45
Epoch: 30/1500, loss: 37.00813626283828, correct: 47
Epoch: 40/1500, loss: 36.53294153636345, correct: 48
Epoch: 50/1500, loss: 36.09904077736577, correct: 47
Epoch: 60/1500, loss: 35.712140331137086, correct: 47
Epoch: 70/1500, loss: 35.35454351529604, correct: 48
Epoch: 80/1500, loss: 34.974734718490325, correct: 48
Epoch: 90/1500, loss: 34.571895265247036, correct: 48
Epoch: 100/1500, loss: 34.153311165131214, correct: 47
Epoch: 110/1500, loss: 33.70523240877096, correct: 47
Epoch: 120/1500, loss: 33.24172884771691, correct: 47
Epoch: 130/1500, loss: 32.7836943460168, correct: 47
Epoch: 140/1500, loss: 32.314791105582906, correct: 51
Epoch: 150/1500, loss: 31.839863086502458, correct: 51
Epoch: 160/1500, loss: 31.371237535507376, correct: 52
Epoch: 170/1500, loss: 30.902191033500436, correct: 52
Epoch: 180/1500, loss: 30.42703641649672, correct: 53
Epoch: 190/1500, loss: 29.945674908916445, correct: 53
Epoch: 200/1500, loss: 29.456998912209805, correct: 54
Epoch: 210/1500, loss: 28.961456966326857, correct: 54
Epoch: 220/1500, loss: 28.46297310047025, correct: 56
Epoch: 230/1500, loss: 27.96077968339974, correct: 56
Epoch: 240/1500, loss: 27.454252613211242, correct: 57
Epoch: 250/1500, loss: 26.946810165888664, correct: 57
Epoch: 260/1500, loss: 26.43724189957142, correct: 57
Epoch: 270/1500, loss: 25.9301333993795, correct: 56
Epoch: 280/1500, loss: 25.424098524666633, correct: 56
Epoch: 290/1500, loss: 24.919883044510332, correct: 57
Epoch: 300/1500, loss: 24.418223219614646, correct: 57
Epoch: 310/1500, loss: 23.92216547597356, correct: 57
Epoch: 320/1500, loss: 23.432533702733657, correct: 57
Epoch: 330/1500, loss: 22.949651310831094, correct: 57
Epoch: 340/1500, loss: 22.47646755460463, correct: 57
Epoch: 350/1500, loss: 22.01343846186734, correct: 57
Epoch: 360/1500, loss: 21.56136891001967, correct: 57
Epoch: 370/1500, loss: 21.11937349874681, correct: 57
Epoch: 380/1500, loss: 20.6884204178444, correct: 57
Epoch: 390/1500, loss: 20.26863323047686, correct: 57
Epoch: 400/1500, loss: 19.86009092373521, correct: 57
Epoch: 410/1500, loss: 19.4628338497699, correct: 57
Epoch: 420/1500, loss: 19.076921746748216, correct: 57
Epoch: 430/1500, loss: 18.70256466277067, correct: 57
Epoch: 440/1500, loss: 18.340070989662777, correct: 57
Epoch: 450/1500, loss: 17.98951432609485, correct: 57
Epoch: 460/1500, loss: 17.65140408333457, correct: 57
Epoch: 470/1500, loss: 17.324864666674895, correct: 57
Epoch: 480/1500, loss: 17.00932537260393, correct: 57
Epoch: 490/1500, loss: 16.70458088578261, correct: 57
Epoch: 500/1500, loss: 16.411976035213378, correct: 57
Epoch: 510/1500, loss: 16.1301305546084, correct: 57
Epoch: 520/1500, loss: 15.858793026223253, correct: 57
Epoch: 530/1500, loss: 15.597493350780056, correct: 57
Epoch: 540/1500, loss: 15.345816551881533, correct: 57
Epoch: 550/1500, loss: 15.10389860931582, correct: 57
Epoch: 560/1500, loss: 14.87135267375678, correct: 57
Epoch: 570/1500, loss: 14.647016248323517, correct: 57
Epoch: 580/1500, loss: 14.430729844336902, correct: 57
Epoch: 590/1500, loss: 14.222629022228897, correct: 57
Epoch: 600/1500, loss: 14.02230152568986, correct: 57
Epoch: 610/1500, loss: 13.82906579048051, correct: 57
Epoch: 620/1500, loss: 13.642949763879367, correct: 57
Epoch: 630/1500, loss: 13.463374218352994, correct: 57
Epoch: 640/1500, loss: 13.289920559095158, correct: 57
Epoch: 650/1500, loss: 13.122403487769231, correct: 57
Epoch: 660/1500, loss: 12.960456953323003, correct: 57
Epoch: 670/1500, loss: 12.803864022495102, correct: 57
Epoch: 680/1500, loss: 12.652437089538715, correct: 57
Epoch: 690/1500, loss: 12.505687542656103, correct: 57
Epoch: 700/1500, loss: 12.363485306813436, correct: 57
Epoch: 710/1500, loss: 12.225827299307612, correct: 57
Epoch: 720/1500, loss: 12.093081037351412, correct: 57
Epoch: 730/1500, loss: 11.96331548052137, correct: 57
Epoch: 740/1500, loss: 11.837806934200902, correct: 57
Epoch: 750/1500, loss: 11.716470338505669, correct: 57
Epoch: 760/1500, loss: 11.598191625327285, correct: 57
Epoch: 770/1500, loss: 11.482883435534678, correct: 57
Epoch: 780/1500, loss: 11.371286656631229, correct: 57
Epoch: 790/1500, loss: 11.262357075289534, correct: 57
Epoch: 800/1500, loss: 11.15744320129301, correct: 57
Epoch: 810/1500, loss: 11.055028338058234, correct: 57
Epoch: 820/1500, loss: 10.95517563446339, correct: 57
Epoch: 830/1500, loss: 10.858092922408977, correct: 57
Epoch: 840/1500, loss: 10.763467364917958, correct: 57
Epoch: 850/1500, loss: 10.671172281942704, correct: 57
Epoch: 860/1500, loss: 10.581175072544116, correct: 57
Epoch: 870/1500, loss: 10.49327707013031, correct: 57
Epoch: 880/1500, loss: 10.407206528730125, correct: 57
Epoch: 890/1500, loss: 10.32244905586124, correct: 57
Epoch: 900/1500, loss: 10.239805085597101, correct: 57
Epoch: 910/1500, loss: 10.159044866974208, correct: 57
Epoch: 920/1500, loss: 10.0807974103657, correct: 57
Epoch: 930/1500, loss: 10.004428051096882, correct: 57
Epoch: 940/1500, loss: 9.929617038777005, correct: 57
Epoch: 950/1500, loss: 9.856256987470362, correct: 57
Epoch: 960/1500, loss: 9.78504896040533, correct: 57
Epoch: 970/1500, loss: 9.714339392527327, correct: 57
Epoch: 980/1500, loss: 9.645844910022284, correct: 57
Epoch: 990/1500, loss: 9.577827396762693, correct: 57
Epoch: 1000/1500, loss: 9.511843229342215, correct: 57
Epoch: 1010/1500, loss: 9.449019844801107, correct: 57
Epoch: 1020/1500, loss: 9.38576356691895, correct: 57
Epoch: 1030/1500, loss: 9.325360718235498, correct: 57
Epoch: 1040/1500, loss: 9.263754161662716, correct: 57
Epoch: 1050/1500, loss: 9.205302453118017, correct: 57
Epoch: 1060/1500, loss: 9.147931430886642, correct: 57
Epoch: 1070/1500, loss: 9.089697873866992, correct: 57
Epoch: 1080/1500, loss: 9.03333639587283, correct: 57
Epoch: 1090/1500, loss: 8.978537960022678, correct: 57
Epoch: 1100/1500, loss: 8.924524051951334, correct: 57
Epoch: 1110/1500, loss: 8.871505545218454, correct: 57
Epoch: 1120/1500, loss: 8.818792621670857, correct: 57
Epoch: 1130/1500, loss: 8.767571910956866, correct: 57
Epoch: 1140/1500, loss: 8.71607076425784, correct: 57
Epoch: 1150/1500, loss: 8.665517296874432, correct: 57
Epoch: 1160/1500, loss: 8.615551596660652, correct: 57
Epoch: 1170/1500, loss: 8.566560888229414, correct: 57
Epoch: 1180/1500, loss: 8.520432515222373, correct: 57
Epoch: 1190/1500, loss: 8.473410042376537, correct: 57
Epoch: 1200/1500, loss: 8.425655258949085, correct: 57
Epoch: 1210/1500, loss: 8.378528080622035, correct: 57
Epoch: 1220/1500, loss: 8.335254723899531, correct: 57
Epoch: 1230/1500, loss: 8.280186777212279, correct: 57
Epoch: 1240/1500, loss: 8.226666433684994, correct: 57
Epoch: 1250/1500, loss: 8.174200580460104, correct: 57
Epoch: 1260/1500, loss: 8.124271020942327, correct: 57
Epoch: 1270/1500, loss: 8.075305551822183, correct: 57
Epoch: 1280/1500, loss: 8.027962286550979, correct: 57
Epoch: 1290/1500, loss: 7.978322749017063, correct: 57
Epoch: 1300/1500, loss: 7.935110471177045, correct: 57
Epoch: 1310/1500, loss: 7.888680932147585, correct: 57
Epoch: 1320/1500, loss: 7.845217146752558, correct: 57
Epoch: 1330/1500, loss: 7.801527089811185, correct: 57
Epoch: 1340/1500, loss: 7.755843987179961, correct: 57
Epoch: 1350/1500, loss: 7.713524446611868, correct: 57
Epoch: 1360/1500, loss: 7.674961543334313, correct: 57
Epoch: 1370/1500, loss: 7.6295794314339584, correct: 57
Epoch: 1380/1500, loss: 7.5898433588780865, correct: 57
Epoch: 1390/1500, loss: 7.5498188467169465, correct: 57
Epoch: 1400/1500, loss: 7.510705967286793, correct: 57
Epoch: 1410/1500, loss: 7.47015380273805, correct: 57
Epoch: 1420/1500, loss: 7.433244032087585, correct: 57
Epoch: 1430/1500, loss: 7.396780411019812, correct: 57
Epoch: 1440/1500, loss: 7.357718327120473, correct: 57
Epoch: 1450/1500, loss: 7.323290330250138, correct: 57
Epoch: 1460/1500, loss: 7.28453608131206, correct: 57
Epoch: 1470/1500, loss: 7.249554406804577, correct: 57
Epoch: 1480/1500, loss: 7.211282685901509, correct: 57
Epoch: 1490/1500, loss: 7.178673354010004, correct: 57
Epoch: 1500/1500, loss: 7.142075681333638, correct: 57
