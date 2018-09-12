'''
    Converts Outputs, Inputs into number representation
    This file has these classes:
        * OUTPUTS
            * PARAM_SITE - site outputs (not used in neural network)
            * ACTION - operator picked actions
            * MEMORY - operator picked memory adjustments
                * original (code is commented)
                * bucketed (used code)
            * XROOTD - is enabled/disabled
            * SPLITTING - operator picked splitting
                * original (code is commented)
                * bucketed (used code)
        * INPUTS
            * SITE - site where error happened
            * ERROR - all error codes
        * NOT IMPORTANT (breaks down site into smaller categories)
            * COUNTRY = SITE[0:2], where SITE is site string
            * TIER = SITE[3:5], where SITE is site string
            * POSITIVITY - good/bad site

    Each class has variables asociated with it:
        * CLASS_NAME2index - converts all possible names to index
        * index2CLASS_NAME - converts number (index) -> class name
        * CLASS_NAME_names - list of names
'''

import numpy as np

'''"""""""""""""""""" OUTPUTS """""""""""""""""'''
# ============================================== #
#                   PARAM_SITE                   #
# ============================================== #

param_site2index = {
'T0_CH_CERN':0,
'T1_DE_KIT':1,
'T1_ES_PIC':2,
'T1_FR_CCIN2P3':3,
'T1_IT_CNAF':4,
'T1_RU_JINR':5,
'T1_UK_RAL':6,
'T1_US_FNAL':7,
'T2_AT_Vienna':8,
'T2_BE_IIHE':9,
'T2_BE_UCL':10,
'T2_BR_SPRACE':11,
'T2_CH_CERN':12,
'T2_CH_CERN_HLT':13,
'T2_CH_CSCS':14,
'T2_CH_CSCS_HPC':15,
'T2_CN_Beijing':16,
'T2_DE_DESY':17,
'T2_DE_RWTH':18,
'T2_EE_Estonia':19,
'T2_ES_CIEMAT':20,
'T2_ES_IFCA':21,
'T2_FI_HIP':22,
'T2_FR_CCIN2P3':23,
'T2_FR_GRIF_IRFU':24,
'T2_FR_GRIF_LLR':25,
'T2_FR_IPHC':26,
'T2_GR_Ioannina':27,
'T2_HU_Budapest':28,
'T2_IN_TIFR':29,
'T2_IT_Bari':30,
'T2_IT_Legnaro':31,
'T2_IT_Pisa':32,
'T2_IT_Rome':33,
'T2_KR_KISTI':34,
'T2_KR_KNU':35,
'T2_PL_Swierk':36,
'T2_PL_Warsaw':37,
'T2_PT_NCG_Lisbon':38,
'T2_RU_IHEP':39,
'T2_RU_INR':40,
'T2_RU_ITEP':41,
'T2_RU_JINR':42,
'T2_TR_METU':43,
'T2_TW_NCHC':44,
'T2_UA_KIPT':45,
'T2_UK_London_Brunel':46,
'T2_UK_London_IC':47,
'T2_UK_SGrid_Bristol':48,
'T2_UK_SGrid_RALPP':49,
'T2_US_Caltech':50,
'T2_US_Florida':51,
'T2_US_MIT':52,
'T2_US_Nebraska':53,
'T2_US_Purdue':54,
'T2_US_UCSD':55,
'T2_US_Vanderbilt':56,
'T2_US_Wisconsin':57,
'T3_CH_CERN_HelixNebula':58,
'T3_UK_London_RHUL':59,
'T3_UK_SGrid_Oxford':60,
'U':61,
'_':61,
'a':61,
'e':61,
'g':61,
'k':61,
'n':61,
'o':61,
'r':61,
'y':61,
'1':61,
'2':61,
'A':61,
'C':61,
'E':61,
'F':61,
'H':61,
'I':61,
'L':61,
'N':61,
'P':61,
'R':61,
'S':61,
'T':61,
'':61,
'key_error':61,
}
max_val = np.max([a for a in param_site2index.values()])
index2param_site = {b:a for a,b in param_site2index.items()}
index2param_site[max_val] = 'default'
param_site_names = list(index2param_site.values())

# ============================================== #
#                     ACTION                     #
# ============================================== #

action2index = {
'acdc':0,
'clone':1,
'special':2,
}
index2action = {b:a for a,b in action2index.items()}
action_names = list(index2action.values())

# ============================================== #
#                     MEMORY                     #
# ============================================== #

# ################# (original) ####################
# memory2index = {
# '10000':0,
# '11000':1,
# '12000':2,
# '14900':3,
# '15000':4,
# '15200':5,
# '16000':6,
# '18000':7,
# '180000':8,
# '19000':9,
# '2000':10,
# '20000':11,
# '20480':12,
# '25000':13,
# '28000':14,
# '3000':15,
# '30000':16,
# '32000':17,
# '4000':18,
# '40000':19,
# '5000':20,
# '6000':21,
# '7000':22,
# '8000':23,
# '9000':24,
# '':25,
# 'key_error':25,
# }
# max_val = np.max([a for a in memory2index.values()])
# index2memory = {b:a for a,b in memory2index.items()}
# index2memory[max_val] = 'default'
# memory_names = list(index2memory.values())

################## (bucketed) ####################
memory2index = {
'2000':0,
'3000':0,
'4000':0,
'5000':1,
'6000':1,
'7000':1,
'8000':1,
'9000':1,
'10000':2,
'11000':2,
'12000':2,
'14900':2,
'15000':2,
'15200':2,
'16000':2,
'18000':3,
'19000':3,
'20000':3,
'20480':3,
'180000':4,
'25000':4,
'28000':4,
'30000':4,
'32000':4,
'40000':4,
'':5,
'key_error':5,
}
index2memory = {
0:'2k-4k',
1:'5k-9k',
2:'10k-16k',
3:'18k-20k',
4:'20k-max',
5:'default'
}
memory_names = list(index2memory.values())


# ============================================== #
#                     XROOTD                     #
# ============================================== #

xrootd2index = {
'disabled':0,
'enabled':1,
'':2,
'key_error':2,
}
max_val = np.max([a for a in xrootd2index.values()])
index2xrootd = {b:a for a,b in xrootd2index.items()}
index2xrootd[max_val] = 'default'
xrootd_names = list(index2xrootd.values())

# ============================================== #
#                   SPLITTING                    #
# ============================================== #

# ################# (original) ####################
# splitting2index = {
# '100x':0,
# '10x':1,
# '20x':2,
# '2x':3,
# '3x':4,
# '50x':5,
# 'max':6,
# '':7,
# 'key_error':7,
# }
# max_val = np.max([a for a in splitting2index.values()])
# index2splitting = {b:a for a,b in splitting2index.items()}
# index2splitting[max_val] = 'default'
# splitting_names = list(index2splitting.values())

################## (bucketed) ####################
splitting2index = {
'100x':1,
'10x':1,
'20x':1,
'2x':0,
'3x':0,
'50x':1,
'max':1,
'':2,
'key_error':2,
}
index2splitting = {
0:'2x-3x',
1:'10x-100x-max',
2:'default'
}
splitting_names = list(index2splitting.values())




'''""""""""""""""""" INPUTS """""""""""""""""""'''

# ============================================== #
#                      SITE                      #
# ============================================== #

site2index = {
'T0_CH_CERN':0,
'T0_CH_CERN_Disk':1,
'T0_CH_CERN_Export':2,
'T0_CH_CERN_MSS':3,
'T1_DE_KIT':4,
'T1_DE_KIT_Disk':5,
'T1_DE_KIT_MSS':6,
'T1_ES_PIC':7,
'T1_ES_PIC_Disk':8,
'T1_ES_PIC_MSS':9,
'T1_FR_CCIN2P3':10,
'T1_FR_CCIN2P3_Disk':11,
'T1_FR_CCIN2P3_MSS':12,
'T1_IT_CNAF':13,
'T1_IT_CNAF_Disk':14,
'T1_IT_CNAF_MSS':15,
'T1_RU_JINR':16,
'T1_RU_JINR_Disk':17,
'T1_RU_JINR_MSS':18,
'T1_UK_RAL':19,
'T1_UK_RAL_Disk':20,
'T1_UK_RAL_ECHO_Disk':21,
'T1_UK_RAL_MSS':22,
'T1_US_FNAL':23,
'T1_US_FNAL_Disk':24,
'T1_US_FNAL_MSS':25,
'T2_AT_Vienna':26,
'T2_BE_IIHE':27,
'T2_BE_UCL':28,
'T2_BR_SPRACE':29,
'T2_BR_UERJ':30,
'T2_CH_CERN':31,
'T2_CH_CERNBOX':32,
'T2_CH_CERN_HLT':33,
'T2_CH_CSCS':34,
'T2_CH_CSCS_HPC':35,
'T2_CN_Beijing':36,
'T2_DE_DESY':37,
'T2_DE_RWTH':38,
'T2_EE_Estonia':39,
'T2_ES_CIEMAT':40,
'T2_ES_IFCA':41,
'T2_FI_HIP':42,
'T2_FR_CCIN2P3':43,
'T2_FR_GRIF_IRFU':44,
'T2_FR_GRIF_LLR':45,
'T2_FR_IPHC':46,
'T2_GR_Ioannina':47,
'T2_HU_Budapest':48,
'T2_IN_TIFR':49,
'T2_IT_Bari':50,
'T2_IT_Legnaro':51,
'T2_IT_Pisa':52,
'T2_IT_Rome':53,
'T2_KR_KISTI':54,
'T2_KR_KNU':55,
'T2_MY_UPM_BIRUNI':56,
'T2_PK_NCP':57,
'T2_PL_Swierk':58,
'T2_PL_Warsaw':59,
'T2_PT_NCG_Lisbon':60,
'T2_RU_IHEP':61,
'T2_RU_INR':62,
'T2_RU_ITEP':63,
'T2_RU_JINR':64,
'T2_RU_PNPI':65,
'T2_RU_SINP':66,
'T2_TH_CUNSTDA':67,
'T2_TR_METU':68,
'T2_TW_NCHC':69,
'T2_UA_KIPT':70,
'T2_UK_London_Brunel':71,
'T2_UK_London_IC':72,
'T2_UK_SGrid_Bristol':73,
'T2_UK_SGrid_RALPP':74,
'T2_US_Caltech':75,
'T2_US_Florida':76,
'T2_US_MIT':77,
'T2_US_Nebraska':78,
'T2_US_Purdue':79,
'T2_US_UCSD':80,
'T2_US_Vanderbilt':81,
'T2_US_Wisconsin':82,
'T3_BG_UNI_SOFIA':83,
'T3_BY_NCPHEP':84,
'T3_CH_CERN_HelixNebula':85,
'T3_CH_PSI':86,
'T3_CH_Volunteer':87,
'T3_CN_PKU':88,
'T3_CO_Uniandes':89,
'T3_ES_Oviedo':90,
'T3_FR_IPNL':91,
'T3_GR_IASA_GR':92,
'T3_GR_IASA_HG':93,
'T3_HU_Debrecen':94,
'T3_IN_PUHEP':95,
'T3_IN_TIFRCloud':96,
'T3_IT_Bologna':97,
'T3_IT_Napoli':98,
'T3_IT_Perugia':99,
'T3_IT_Trieste':100,
'T3_KR_KNU':101,
'T3_KR_UOS':102,
'T3_MX_Cinvestav':103,
'T3_RU_FIAN':104,
'T3_TW_NCU':105,
'T3_TW_NTU_HEP':106,
'T3_UK_London_QMUL':107,
'T3_UK_London_RHUL':108,
'T3_UK_London_UCL':109,
'T3_UK_SGrid_Oxford':110,
'T3_UK_ScotGrid_GLA':111,
'T3_US_Baylor':112,
'T3_US_Colorado':113,
'T3_US_Cornell':114,
'T3_US_FIT':115,
'T3_US_FIU':116,
'T3_US_FNALLPC':117,
'T3_US_FSU':118,
'T3_US_JHU':119,
'T3_US_Kansas':120,
'T3_US_MIT':121,
'T3_US_NERSC':122,
'T3_US_NU':123,
'T3_US_NotreDame':124,
'T3_US_OSG':125,
'T3_US_OSU':126,
'T3_US_Omaha':127,
'T3_US_Princeton_ICSE':128,
'T3_US_PuertoRico':129,
'T3_US_Rice':130,
'T3_US_Rutgers':131,
'T3_US_SDSC':132,
'T3_US_TAMU':133,
'T3_US_TTU':134,
'T3_US_UCD':135,
'T3_US_UCR':136,
'T3_US_UCSB':137,
'T3_US_UMD':138,
'T3_US_UMiss':139,
'T3_US_Vanderbilt_EC2':140,
'Unknown':141,
'null':141,
'NoReportedSite':141,
}
max_val = np.max([a for a in site2index.values()])
index2site = {b:a for a,b in site2index.items()}
index2site[max_val] = 'default'
site_names = list(index2splitting.values())

# ============================================== #
#                     ERROR                      #
# ============================================== #

error2index = {
'-1':0,
'1':1,
'11003':2,
'132':3,
'134':4,
'135':5,
'136':6,
'137':7,
'139':8,
'143':9,
'50110':10,
'50115':11,
'50513':12,
'50660':13,
'50661':14,
'50664':15,
'53':16,
'60307':17,
'60405':18,
'60450':19,
'61202':20,
'70':21,
'70318':22,
'70452':23,
'71':24,
'71101':25,
'71102':26,
'71104':27,
'71302':28,
'71303':29,
'71304':30,
'71305':31,
'73':32,
'75':33,
'76':34,
'80':35,
'8001':36,
'8002':37,
'8003':38,
'8004':39,
'8021':40,
'8026':41,
'84':42,
'85':43,
'86':44,
'87':45,
'92':46,
'99109':47,
'99303':48,
'99305':49,
'99400':50,
'99401':51,
'99996':52,
'99999':53,
}
index2error = {b:a for a,b in error2index.items()}
error_names = list(index2error.values())





'''""""""""""""""" NOT IMPORTANT """""""""""""""'''

# ============================================== #
#                    COUNTRY                     #
# ============================================== #

country2index = {
'CH':0,
'DE':1,
'ES':2,
'FR':3,
'IT':4,
'RU':5,
'UK':6,
'US':7,
'AT':8,
'BE':9,
'BR':10,
'CN':11,
'EE':12,
'FI':13,
'GR':14,
'HU':15,
'IN':16,
'KR':17,
'MY':18,
'PK':19,
'PL':20,
'PT':21,
'TH':22,
'TR':23,
'TW':24,
'UA':25,
'BG':26,
'BY':27,
'CO':28,
'MX':29,
'no':30,
'l':30,
'ep':30,
'':30,
}
max_val = np.max([a for a in country2index.values()])
index2country = {b:a for a,b in country2index.items()}
index2country[max_val] = 'unknown'
country_names = list(index2country.values())

# ============================================== #
#                      TIER                      #
# ============================================== #

tier2index = {
'T0':0,
'T1':1,
'T2':2,
'T3':3,
'Un':4,
'nu':4,
'No':4,
'':4,
}
max_val = np.max([a for a in tier2index.values()])
index2tier = {b:a for a,b in tier2index.items()}
index2tier[max_val] = 'unknown'
tier_names = list(index2tier.values())

# ============================================== #
#                 POSITIVITY                     #
# ============================================== #

positivity2index = {
'good_sites':0,
'bad_sites':1,
}
index2positivity = {b:a for a,b in positivity2index.items()}
positivity_names = list(index2positivity.values())