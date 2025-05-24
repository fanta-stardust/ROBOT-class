import numpy as np


def load_tag_pos():
    def expand(LL):
        z_tag = np.array([0])
        NN = np.append(LL, z_tag)
    # 二维码四个角顶点坐标关系（在XZ或YZ平面内）
        a = np.array([
            [0, 0, 0], [0, 5, 0], [5, 5, 0], [5, 0, 0]
        ],dtype=np.float64)
        return a + NN

    x_negative = ["1","7","10","13","19","21","26","29","35"]
    y_negative = ["4","6","9","14","18","22","28","32","36"]
    x_positive = ["3","5","12","16","17","23","27","31","33"]
    y_positive = ["2","8","11","15","20","24","25","30","34"]
    x_neg_shift = np.array([[0, 0, 0],  [0, -5, 0], [0, -5, -5],[0, 0, -5]])
    y_neg_shift = np.array([[0, 0, 0],[5, 0, 0], [5, 0, -5], [0, 0, -5]])
    x_pos_shift = np.array([[0, 0, 0],[0, 5, 0], [0, 5, -5], [0, 0, -5]])
    y_pos_shift = np.array([[0, 0, 0],  [-5, 0, 0], [-5, 0, -5],[0, 0, -5]])
    
    def expand_2(i,MM):
        z_tag = np.array([39.8])
        NN = np.append(MM, z_tag)
        if i in x_negative:
            return x_neg_shift + NN
        elif i in y_negative:
            return y_neg_shift + NN
        elif i in x_positive:
            return x_pos_shift + NN
        elif i in y_positive:
            return y_pos_shift + NN
        else:
            print("该点不在列表中！")

    tag_poses = {}
    tag_poses['37'] = expand([166, 287])
    tag_poses['38'] = expand([231.3, 287.5])
    tag_poses['39'] = expand([9.5, 250])
    tag_poses['40'] = expand([48.7, 266])
    tag_poses['41'] = expand([279.5, 255])
    tag_poses['42'] = expand([278, 182])
    tag_poses['43'] = expand([271.5, 36])
    tag_poses['44'] = expand([263, 108])
    tag_poses['45'] = expand([225, 126])
    tag_poses['46'] = expand([226.1, 62])
    tag_poses['47'] = expand([114.2, 60])
    tag_poses['48'] = expand([65.5, 23])
    tag_poses['49'] = expand([22.5, 24.5])
    tag_poses['50'] = expand([80, 95])
    tag_poses['51'] = expand([12.4, 142])
    tag_poses['52'] = expand([12.5, 187])
    tag_poses['53'] = expand([126, 237])
    tag_poses['54'] = expand([143, 141])
    tag_poses['55'] = expand([80.5, 148]) 
    tag_poses['56'] = expand([164, 54.4]) 
    tag_poses['57'] = expand([239, 213.5]) 
    tag_poses['58'] = expand([205, 241]) 
    tag_poses['59'] = expand([199.5, 141.5]) 
    tag_poses['60'] = expand([78.5, 236.5]) 
    tag_poses['61'] = expand([122, 156.5]) 

# 以下是⽩⾊障碍物的坐标，每四个⼀组，共9组，每组的第⼀个对应左上⾓，
# 之后三个从上往下看逆时针⽅向
    tag_poses['1'] = expand_2('1',[196, 10])
    tag_poses['4'] = expand_2('4',[216, 5])
    tag_poses['3'] = expand_2('3',[221, 25])
    tag_poses['2'] = expand_2('2',[201, 30])

    tag_poses['7'] = expand_2('7',[116.5, 11.5])
    tag_poses['6'] = expand_2('6',[136.5, 6.5])
    tag_poses['5'] = expand_2('5',[141.5, 26.5])
    tag_poses['8'] = expand_2('8',[121.5, 31.5])

    tag_poses['10'] = expand_2('10',[14.5, 59.5])
    tag_poses['9'] = expand_2('9',[34.5, 54.5])
    tag_poses['12'] = expand_2('12',[39.5, 74.5])
    tag_poses['11'] = expand_2('11',[19.5, 79.5])

    tag_poses['13'] = expand_2('13',[69.5, 186])
    tag_poses['14'] = expand_2('14',[89.5, 181])
    tag_poses['16'] = expand_2('16',[94.5, 201])
    tag_poses['15'] = expand_2('15',[74.5, 206])

    tag_poses['19'] = expand_2('19',[92, 268])
    tag_poses['18'] = expand_2('18',[112, 263])
    tag_poses['17'] = expand_2('17',[117, 283])
    tag_poses['20'] = expand_2('20',[97, 288])

    tag_poses['21'] = expand_2('21',[154.5, 189])
    tag_poses['22'] = expand_2('22',[174.5, 184])
    tag_poses['23'] = expand_2('23',[179.5, 204])
    tag_poses['24'] = expand_2('24',[159.5, 209])

    tag_poses['26'] = expand_2('26',[235. , 251.5])
    tag_poses['28'] = expand_2('28',[255. , 246.5])
    tag_poses['27'] = expand_2('27',[260. , 266.5])
    tag_poses['25'] = expand_2('25',[240. , 271.5])

    tag_poses['29'] = expand_2('29',[235.5, 150.8])
    tag_poses['32'] = expand_2('32',[255.5, 145.8])
    tag_poses['31'] = expand_2('31',[260.5, 165.8])
    tag_poses['30'] = expand_2('30',[240.5, 170.8])

    tag_poses['35'] = expand_2('35',[158, 90])
    tag_poses['36'] = expand_2('36',[178, 85])
    tag_poses['33'] = expand_2('33',[183, 105])
    tag_poses['34'] = expand_2('34',[163, 110])
    return tag_poses