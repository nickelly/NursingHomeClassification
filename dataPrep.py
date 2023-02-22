import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

#perform aggregating and joining of data
def prepJoinedNursingHomeData():

    #load datasets, remove spaces from column names
    citationsDF = pd.read_csv('datasets/NH_HealthCitations_Jan2023.csv', dtype = {'Federal Provider Number' : str})
    citationsDF.columns = citationsDF.columns.str.replace(' ', '')
    penaltysDF = pd.read_csv('datasets/NH_Penalties_Jan2023.csv')
    penaltysDF.columns = penaltysDF.columns.str.replace(' ', '')
    medicareClaimsDF = pd.read_csv('datasets/NH_QualityMsr_Claims_Jan2023.csv')
    medicareClaimsDF.columns = medicareClaimsDF.columns.str.replace(' ', '')
    residentAssessmentDF = pd.read_csv('datasets/NH_QualityMsr_MDS_Jan2023.csv', dtype = {'Federal Provider Number': str})
    residentAssessmentDF.columns = residentAssessmentDF.columns.str.replace(' ', '')
    careMeasuresDF = pd.read_csv('datasets/Skilled_Nursing_Facility_Quality_Reporting_Program_Provider_Data_Jan2023.csv' , dtype = {'CMS Certification Number (CCN)': str})
    careMeasuresDF.columns = careMeasuresDF.columns.str.replace(' ', '')

    #get count of citations by federal provider number
    citationCategories = ['StandardDeficiency', 'ComplaintDeficiency', 'InfectionControlInspectionDeficiency', 'CitationunderIDR', 'CitationunderIIDR']
    citationsDF[citationCategories] = citationsDF[citationCategories].astype(str).replace('N', np.NaN)
    citationsCounts = pd.DataFrame(citationsDF.groupby('FederalProviderNumber')[citationCategories].count())
    
    #get count of fines by federal provider number
    fineCounts = pd.DataFrame((penaltysDF.loc[penaltysDF["PenaltyType"] == "Fine"]).groupby("FederalProviderNumber")["PenaltyType"].count())
    paymentDenialCounts = pd.DataFrame((penaltysDF.loc[penaltysDF["PenaltyType"] == "Payment Denial"]).groupby("FederalProviderNumber")["PenaltyType"].count())
    fineCounts.rename(columns = {"PenaltyType": "fineCounts"}, inplace=True)
    paymentDenialCounts.rename(columns = {"PenaltyType": "paymentDenialCounts"}, inplace=True)
    penaltyCounts = fineCounts.join(paymentDenialCounts, how = "outer")

    #go from one row per evaluation metric to one column per metric by federal provider number
    columnLabels = ["FederalProviderNumber", "AdjustedScore", "ExpectedScore"]
    newColumnLabels = ["AdjustedScore", "ExpectedScore"]
    codes = medicareClaimsDF.MeasureCode.unique()
    codeList = [0,1,2,3]
    for index in range(len(codes)):
        codeList[index] = medicareClaimsDF[columnLabels].loc[medicareClaimsDF["MeasureCode"] == codes[index]]
        codeList[index].columns = ["FederalProviderNumber"] + [label + str(codes[index]) for label in newColumnLabels]
        codeList[index].set_index("FederalProviderNumber", inplace = True)
    
    #go from one row per evaluation metric to one column per metric by federal provider number
    medicareCodes = (pd.DataFrame(medicareClaimsDF.FederalProviderNumber.unique(), columns = {"FederalProviderNumber"})).set_index("FederalProviderNumber")
    medicareCodes = medicareCodes.join(codeList)
    columnLabels = ["FederalProviderNumber", "Q1MeasureScore", "Q2MeasureScore", "Q3MeasureScore", "Q4MeasureScore"]
    newColumnLabels = ["Q1MeasureScore", "Q2MeasureScore", "Q3MeasureScore", "Q4MeasureScore"]
    codes = residentAssessmentDF.MeasureCode.unique()
    codeList = list(range(len(codes)))
    for index in range(len(codes)):
        codeList[index] = residentAssessmentDF[columnLabels].loc[residentAssessmentDF["MeasureCode"] == codes[index]]
        codeList[index].columns = ["FederalProviderNumber"] + [label + str(codes[index]) for label in newColumnLabels]
        codeList[index].set_index("FederalProviderNumber", inplace = True)
    
    #go from one row per evaluation metric to one column per metric by federal provider number
    residentialAssessmentCodes = (pd.DataFrame(residentAssessmentDF.FederalProviderNumber.unique(), columns = {"FederalProviderNumber"})).set_index("FederalProviderNumber")
    residentialAssessmentCodes = residentialAssessmentCodes.join(codeList)
    careScores = [0,0]
    careScores[0] = careMeasuresDF[["CMSCertificationNumber(CCN)", "Score"]].loc[careMeasuresDF["MeasureCode"] == "S_005_02_DTC_COMP_PERF"]
    careScores[1] = careMeasuresDF[["CMSCertificationNumber(CCN)", "Score"]].loc[careMeasuresDF["MeasureCode"] == "S_039_01_HAI_COMP_PERF"]
    careScores[0].columns = ["FederalProviderNumber", "FacilityReadmissionScore"]
    careScores[1].columns = ["FederalProviderNumber", "InfectionScore"]
    careScores[0].set_index("FederalProviderNumber", inplace = True)
    careScores[1].set_index("FederalProviderNumber", inplace = True)
    careScoresDF = (pd.DataFrame(careMeasuresDF["CMSCertificationNumber(CCN)"].unique(), columns = {"FederalProviderNumber"})).set_index("FederalProviderNumber")
    careScoresDF = careScoresDF.join(careScores)

    #join all the datasets and return
    nursingHomeData = careScoresDF.join([residentialAssessmentCodes, medicareCodes, penaltyCounts, citationsCounts], how="outer")
    return nursingHomeData


#returns train/test data split into features and labels in format: trainX, trainY, testX, testY
#target can be "FacilityReadmissionScore" or "InfectionScore"
#scaler = "minmax" or "standard"
def preprocessData(data, splitSeed = 0, target = 'FacilityReadmissionScore', doImpute=True, doPCA = True, scalerType="standard"):
    
    #drop rows without a target
    data.drop((data.loc[(data[target] == "Not Available")]).index, inplace = True)
    
    #split data into train and test
    trainDF, testDF = train_test_split(data, test_size = 0.3, random_state = splitSeed)
    
    #the count based column names saved to a list
    countsColumns = ["fineCounts", "paymentDenialCounts", "StandardDeficiency", "ComplaintDeficiency", "InfectionControlInspectionDeficiency", "CitationunderIDR", "CitationunderIIDR"]
    
    #replace count based nulls with zeroes
    trainDF[countsColumns] = trainDF[countsColumns].fillna(0)
    testDF[countsColumns] = testDF[countsColumns].fillna(0)

    #as we are using KNN imputing, we should scale numeric values so distance measures are comparable
    #if we are optionally performing PCA, we should use standard scaler so variances are comparable between features
    if (doPCA) or (scalerType == "standard"):
        numerics = trainDF.select_dtypes(include='float64').columns
        scaler = StandardScaler()
        scaler.fit(trainDF[numerics])
        trainDF[numerics] = scaler.transform(trainDF[numerics])
        testDF[numerics] = scaler.transform(testDF[numerics])
        if scalerType == "minmax":
            print("Standard scaler used over min-max to perform PCA, to perform min-max scaling set doPCA=False")

    #min max scaler for use with distance based models
    if (scalerType == "minmax") & (not doPCA):
        numerics = trainDF.select_dtypes(include='float64').columns
        scaler = MinMaxScaler()
        scaler.fit(trainDF[numerics])
        trainDF[numerics] = scaler.transform(trainDF[numerics])
        testDF[numerics] = scaler.transform(testDF[numerics])

    if doImpute:
        #impute remaining NaN values usingKNN
        noNA = (trainDF.isnull().sum() == 0).tolist()
        hasNA = np.logical_not(noNA)
        KNNimp = KNNImputer(n_neighbors=2)
        KNNimp.fit(trainDF.loc[:,hasNA])
        trainDF.loc[:,hasNA] = KNNimp.transform(trainDF.loc[:,hasNA])
        testDF.loc[:,hasNA] = KNNimp.transform(testDF.loc[:,hasNA])

    if doPCA:
        #Perform PCA, separate features and targets
        pca = PCA(n_components=0.80)
        pca.fit(trainDF[numerics])
        trainX = pca.transform(trainDF[numerics])
        testX = pca.transform(testDF[numerics])
        trainY = trainDF[target]
        testY = testDF[target]
    else:
        trainX = trainDF[numerics]
        testX = testDF[numerics]
        trainY = trainDF[target]
        testY = testDF[target]

    return trainX, trainY, testX, testY