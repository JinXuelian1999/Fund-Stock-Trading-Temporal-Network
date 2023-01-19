import pandas as pd
from selenium.webdriver import Firefox
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time


def pre_process(filename1, filename2):
    data = pd.read_csv(filename1, low_memory=False)
    data['股票代码'] = data['股票代码'].str[:6]
    # print(data.iloc[3491079])
    data['股票代码'] = data['股票代码'].str.replace("gp_Add", "")
    # print(data.iloc[3491079])

    data.to_csv(filename2, index=False)


def process(filename):
    data = pd.read_csv(filename, low_memory=False)
    # print(data.iloc[3491079])
    stocks = list(data['股票名称'])
    codes = list(data['股票代码'])
    st_to_code = dict(zip(stocks, codes))
    # print(data[data.isnull().T.any()])
    null = data[data.isnull().T.any()]['股票名称'].unique()
    print("有%d处股票代码空白。" % data['股票代码'].isnull().sum())
    print("有%d支股票没有股票代码。" % len(null))
    return st_to_code, null


def crawler(st_name):
    """爬取股票代码"""
    web = Firefox()
    web.get("https://cn.bing.com/")
    time.sleep(1)
    web.find_element(by=By.XPATH, value='//*[@id="sb_form_q"]').send_keys(st_name + '股票代码', Keys.ENTER)
    time.sleep(1)
    try:
        code = web.find_element(by=By.XPATH, value='//div[@class="b_focusTextLarge"]').text
        if len(code) == 6:
            global kz_code
            kz_code[k] = int(code)
            print(kz_code)
        web.close()
    except:
        web.close()


# # kz_code0 = {'腾讯控股': 700, '舜宇光学科技': 2382, '周大福': 1929, '安踏体育': 2020, '周黑鸭': 1458, '敏华控股': 1999,
# #             '锦欣生殖': 1951, '友邦保险': 1299, '美东汽车': 1268, '中升控股': 881, '金山软件': 3888, '中芯国际': 688981,
# #             '中国重汽': 951, '华润燃气': 1193, '新濠国际发展': 200, '海底捞': 6862, '世茂集团': 600823, '融创中国': 1918,
# #             '建设银行': 601939, '石药集团': 1093, '万洲国际': 288, '五矿资源': 831, '理文造纸': 2314, '玖龙纸业': 2689,
# #             '紫金矿业': 601899, '中国铁塔': 788, '银河娱乐': 27, '金沙中国有限公司': 1928, '申洲国际': 2313, '中国神华': 601088,
# #             '中航科工': 2357, '华润水泥控股': 1313, '中国石油化工股份': 386, '三生制药': 1530, '吉利汽车': 175, '敏实集团': 425,
# #             '永利澳门': 1128, '小天鹅A': 418, '招商地产': 24}
# # kz_code1 = {'青海互助青稞酒股份有限公司': 2646, '安徽省凤形耐磨材料股份有限公司': 2760, '内蒙古蒙草生态环境(集团)股份有限公司': 300355,
# #             '保利发展控股集团股份有限公司': 600048, '上海柴油机股份有限公司': 600841, '中国工商银行股份有限公司': 601398,
# #             '比亚迪股份': 2594, '美团-W': 3690, '比亚迪电子': 285, 'JS环球生活': 1691, '金山软件': 3888, '中国生物制药': 1177,
# #             '旭辉控股集团': 884, 'ASM PACIFIC': 522, '中国南方航空股份': 600029, '申洲国际': 2313, '华润电力': 836,
# #             '华润水泥控股': 1313, '中国石油化工股份': 600028, '澳博控股': 880, '诺诚健华-B': 9969, '保利发展': 600048, '*ST金正': 2470,
# #             '中航动力': 600893, '*ST山煤': 600546, '*ST神火': 933, '*ST新集': 601918, '*ST煤气': 968, '中海集运': 601866,
# #             '*ST韶钢': 717, '中国远洋': 601919}
# # kz_code2 = {'信义光能': 968, '旭辉永升服务': 1995, '碧桂园服务': 6098, '香港交易所': 13999, '思摩尔国际': 6969, '中国飞鹤': 6186,
# #             '猫眼娱乐': 1896, '龙湖集团': 960, '海螺创业': 586, '时代中国控股': 1233, '龙光集团': 3380, '江西铜业股份': 600362,
# #             '中电控股': 2, '华润置地': 1109, '中国建材': 3323, '美的电器': 527, '康得退': 2450, '华兴资本控股': 1911,
# #             '众安在线': 6060, '青岛啤酒股份': 600600, '百威亚太': 1876, '药明生物': 2269, '京东健康': 6618, '阿里健康': 241,
# #             '威高股份': 1066, '海尔电器': 600690, '中国东方教育': 667, '中国民航信息网络': 696, '信达生物': 1801, '沛嘉医疗-B': 9996,
# #             '先声药业': 2096, '中金公司': 600489, '中国太保': 601601, '泡泡玛特': 9992, '*ST信威': 600485, '中弘退': 979,
# #             '中国北车': 601299, '武钢股份': 600005, '邯郸钢铁': 600001, '营口港': 600317, '扬子石化': 866, '*ST上航': 600591,
# #             '外运发展': 600270, '海信家电': 921, '退市保千': 600074, '退市华业': 600240, '退市海润': 600401, '宏源证券': 562,
# #             '百联股份': 600827, '退市锐电': 601558, '*ST二重': 601268, '退市长油': 600087, '莱钢股份': 600102,
# #             '退市美都': 600175, '退市刚泰': 600687, '退市工新': 600701, '金亚退': 300028, '天茂退': 2509, '胜景山河': 2525,
# #             '时代电气': 688187, '中海油田服务': 2883, '盛运退': 300090, '广汽长丰': 600991, '千山退': 300216, '中国移动': 600941,
# #             '深赤湾A': 22, '退市鹏起': 600614, '*ST宜生': 600978, '天翔退': 300362, '中国海外发展': 688, '退市金钰': 600086,
# #             '印纪退': 2143, '华虹半导体': 1347, '上药转换': 600849, 'S兰铝': 600296, '上实医药': 600607, 'S*ST云大': 600181,
# #             '江南嘉捷': 601313, '中国电信': 601728, '包头铝业': 600472, '瑞声科技': 2018, 'ESR': 1821, '中国光大控股': 165,
# #             '北讯退': 2359, '华泽退': 693, '龙力退': 2604, '金马集团': 602, '中航善达': 43, '石油大明': 406, '大华农': 300186,
# #             '中国宏桥': 1378, '康基医疗': 9997, '金斯瑞生物科技': 1548, '康健国际医疗': 3886, '香港中华煤气': 3, '融创服务': 1516,
# #             '恒基地产': 12, "VITASOY INT'L": 345, '信义玻璃': 868, '汇丰控股': 5, '达利食品': 3799, '建滔集团': 148, '深圳国际': 152,
# #             '九龙仓集团': 4, '建滔积层板': 1888, 'H&H国际控股': 1112, '九龙仓置业': 1997, '远洋集团': 3377, '富力地产': 2777,
# #             'VTECH HOLDINGS': 303, '耐世特': 1316, '远东宏信': 3360, '海天国际': 1882, '江苏宁沪高速公路': 600377,
# #             '德昌电机控股': 179, '中国通信服务': 552, '华能新能源': 600011, '上海实业控股': 363, '绿叶制药': 2186, '越秀地产': 123,
# #             '中国光大银行': 601818, 'SOHO中国': 410, '海丰国际': 1308, '中银航空租赁': 2588, '鹰君': 41, '富智康集团': 2038,
# #             '南海控股': 680, '港华燃气': 1083, '嘉华国际': 173, '中国联塑': 2128, '湾区发展': 737, '光启科学': 439, '徽商银行': 3698,
# #             '中国天然气': 931, '中国电力': 2380, '北控水务集团': 371, '黛丽斯国际': 333, '嘉里建设': 683, '第一拖拉机股份': 601038,
# #             '高鑫零售': 6808, '重庆钢铁股份': 601005, '太古股份公司B': 87, '同程旅行': 780, '永达汽车': 3669, '裕元集团': 551,
# #             '光大环境': 257, '北京控股': 392, '云顶新耀-B': 1952, '世茂服务': 873, '华润医药': 3320, '中裕燃气': 3633,
# #             '中梁控股': 2772, '美的置业': 3990, '祖龙娱乐': 9990, '盈大地产': 432, '丰盛控股': 607, '华人置业': 127, '利丰': 494,
# #             '晶苑国际': 2232, '民银资本': 1141, '卜蜂国际': 43, '金利丰金融': 1031, '立立电子': 2257, '长江基建集团': 1038,
# #             '华润万象生活': 1209, '百济神州': 688235, '东方锅炉': 600786, '承德钒钛': 600357, '盐湖集团': 792, '退市秋林': 600891,
# #             '昊海生物科技': 688366, '欣泰退': 300372, '国恒退': 594, '路桥建设': 600263, '太行水泥': 600553, '退市博元': 600656,
# #             '欧浦退': 2711, '恒久科技': 2808, '创科实业': 669, '光正教育': 6068, '国美电器': 493, '山东新华制药股份': 756,
# #             '诺辉健康-B': 6606, '天方药业': 600253, '洛阳玻璃股份': 600876, '太古地产': 1972, '*ST上普': 600680, '新都退': 33,
# #             '辽河油田': 817, '上电股份': 600627, '台泥国际集团': 1136, '汇量科技': 1860, '锦州石化': 763, '新湖创业': 600840,
# #             '*ST创智': 787, '*ST信联': 600899, '攀渝钛业': 515, '中西药业': 600842, '吉林化工': 618, 'ST东北高': 600003,
# #             'OKURA HOLDINGS': 1655, '合生元': 1112, '华联商厦': 882, '银泰商业': 1833, '百丽国际': 1880, '*ST成城': 600247,
# #             'S*ST托普': 583}
#
#
# # data1['股票代码'] = data1.apply(lambda x: kz_code[x['股票名称']] if x['股票代码'] == "Nan" else x['股票代码'], axis=1)
# data['股票代码'] = data['股票名称'].map(sc_to_code)
# kz = data[data.isnull().T.any()]['股票名称'].unique()
# print(len(kz))
# kz_code = dict()

#
# sc_to_code = {**sc_to_code, **kz_code}
# data['股票代码'] = data['股票名称'].map(sc_to_code)
# print(data['股票代码'].isnull().sum())
# print(len(data[data.isnull().T.any()]['股票名称'].unique()))


def add_english_name(filename0, filename1):
    """将中文的基金类型和股票行业映射为英文的"""
    # 读取文件
    industries = pd.read_csv(filename0)
    types = pd.read_csv(filename1)
    # 构造字典
    inds = industries["Nindnme"].unique()
    eninds = ['banking', 'real estate', 'comprehensive', 'computer application services', 'other social services',
              'transportation equipment manufacturing industry', 'civil engineering construction industry',
              'manufacture of non-metallic mineral products',
              'wholesale of energy, materials and machinery and electronic equipment', 'information technology industry',
              'buildings completing', 'wholesale and retail trade', 'water transport',
              'real estate development and operation', 'retail industry',
              'the production and supply of electricity, steam and hot water', 'hotel industry', 'metal product industry',
              'other financial industry', 'food processing industry', 'electrical machinery and equipment manufacturing',
              'petroleum processing and coking industry', 'smelting and pressing of non-ferrous metals',
              'professional and scientific research services', 'highway transport industry', 'air transport industry',
              'health, health care, nursing services', 'pharmaceutical industry', 'radio, film and television',
              'special equipment manufacturing', 'securities and futures industry', 'chemical fiber manufacturing industry',
              'general machinery manufacturing', 'extraction of petroleum and natural gas', 'gas production and supply industry',
              'manufacture of raw chemical materials and chemical products', 'leasing Service industry',
              'non-ferrous metals mining and dressing', 'public facilities service', 'paper making and paper products industry',
              'communications and related equipment manufacturing', 'the production and supply of tap water',
              'mining and washing of coal industry', 'railway transportation', 'beverage manufacturing industry',
              'smelting and pressing of ferrous metals', 'other manufacturing industries', 'fishery', 'petroleum, chemicals, plastics',
              'forestry', 'publishing', 'textile industry', 'insurance industry', 'ferrous metal mining industry',
              'furniture manufacturing industry', 'art industry', 'printing industry',
              'agriculture, forestry, animal husbandry, fishery services', 'food manufacturing industry',
              'catering industry', 'animal husbandry', 'communication service industry',
              'wood processing and bamboo, rattan, brown, grass products industry', 'agricultural',
              'traffic and transportation auxiliary industry', 'manufacturing of clothing and other fiber products',
              'instrumentation and cultural, office machinery manufacturing', 'postal service industry',
              'extractive service industry', 'culture, education and sporting goods manufacturing industry',
              'warehousing industry', 'manufacturing of leather, fur, eiderdown and products',
              'other communication and cultural services', 'non-metallic mining industry']
    eninds = [i.capitalize() for i in eninds]
    ind2english = dict(zip(inds, eninds))

    ts = types['基金类型'].unique()
    ents = ['FOF | Medium to high risk', 'Stock | High risk', 'Exchange traded | High risk',
            'Hybrid-partial stock | Medium to high risk', 'Hybrid-Flexible | Medium to high risk',
            'QDII | High risk', 'Hybrid-partial stock', 'Hybrid-partial debt | Medium to high risk',
            'Hybrid-partial debt | Medium risk', 'Mixed-balanced | Medium to high risk', 'Hybrid-flexible | Medium risk',
            'Bond type-hybrid bond | Medium risk', 'Bond-type-hybrid securities', 'Hybrid-flexible',
            'Hybrid-partial debt', 'Commodities(excluding QDII) | High risk', 'FOF | Medium risk',
            'Bond type-long bond | Low and medium risk', 'Hybrid-flexible | High risk',
            'Bond type-hybrid bond | Low and medium risk', 'Hybrid-partial debt | Low and medium risk', 'QDII',
            'Bond type-long bond', 'Bond type-short or medium term loan | Low and medium risk',
            'Bond type-short or medium term loan', 'QDII | Medium risk', 'QDII | Medium to high risk',
            'Monetary  | Low risk', 'Monetary', 'Exchange-Traded', 'Commodities(excluding QDII)',
            'FOF', 'Hybrid-partial stock | Medium risk', 'Bond type-convertible bond | Medium risk',
            'Mixed-absolute returns | Medium to high risk', 'Stock', 'Stock | Medium to high risk',
            'Mixed-absolute returns | Medium risk', 'FOF | High risk', 'Bond type-convertible bond',
            'Reits | High risk', 'Mixed-balanced | Medium risk', 'Mixed-absolute returns',
            'Bond type-long bond | Medium risk', 'Financing type | Low risk', 'Financing type',
            'Bond type-convertible bond | Medium to high risk', 'Reits', 'Mixed-balanced', 'FOF | Low and medium risk',
            'Bond type-convertible bond | Low and medium risk', 'Bond type-hybrid bond | Medium to high risk',
            'Bond type-long bond | Medium to high risk']
    type2english = dict(zip(ts, ents))

    industries["enNindnme"] = industries["Nindnme"].map(ind2english)
    types["enType"] = types["基金类型"].map(type2english)
    industries.to_csv("En_TRD_Co.csv", index=False)
    types.to_csv("En_JJ_Types.csv", index=False)


if __name__ == "__main__":
    # pre_process('JJCC_hebin_gmjj_Fin副本.csv', 'JJCC_hebin_gmjj_Fin副本1.csv')
    # sc_to_code, kz = process('JJCC_hebin_gmjj_Fin副本1.csv')
    # print(kz)
    # kz_code = dict()
    # for k in kz:
    #     crawler(k)
    # print("Over!")
    # sc_to_code = {**sc_to_code, **kz_code}
    # df = pd.read_csv('JJCC_hebin_gmjj_Fin副本1.csv', low_memory=False)
    # df['股票代码'] = df['股票名称'].map(sc_to_code)
    # print("有%d处股票代码空白。" % df['股票代码'].isnull().sum())
    # print("有%d支股票没有股票代码。" % len(df[df.isnull().T.any()]['股票名称'].unique()))
    # df.to_csv('JJCC_hebin_gmjj_Fin副本2.csv', index=False)
    # sc_to_code1, kz1 = process('JJCC_hebin_gmjj_Fin副本2.csv')
    add_english_name("TRD_Co.csv", "基金名称-基金类型.csv")
