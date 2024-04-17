import pandas as pd
import numpy as np
from typing import Union, List
# TODO : addr_state 변수 설정해야한다
class Preprocessor:
    def __init__(self):
        self.file_path = ""
        self.folder_path = ""
        self.df = pd.DataFrame()
    
    def __init__(self, data_file_path:str="", folder_path:str=""):
        self.file_path = data_file_path
        self.folder_path = folder_path
        self.df = pd.DataFrame()
        
    def load_origin_file(self, file_path:str)->None:
        if file_path!="":
            self.file_path = file_path
        self.df = pd.read_csv(self.file_path)
        
    def drop_columns(self, drop_columns_file_path:str = "drop_columns_0402.txt")->None:
        with open(drop_columns_file_path, mode='r') as f:
            drop_fields = f.readlines()
            drop_fields = [drop_field.strip('\n') for drop_field in drop_fields]
        self.df.drop(columns=drop_fields, inplace=True)
    
    def __preprocess_target_variable(self, target_variable:str="loan_status")->None:
        # loan_status가 "current", "issued", "policy" 인 행을 필터링하여 삭제
        modified_df = self.df[~self.df[target_variable].isin(['Current', 'Issued', 'Does not meet the credit policy. Status:Fully Paid', 'Does not meet the credit policy. Status:Charged Off'])]
        # risk = 1, safe = 0 으로 처리
        modified_df.loc[modified_df['loan_status'].isin(['Fully Paid', 'In Grace Period']), 'loan_status'] = 0
        modified_df.loc[modified_df['loan_status'].isin(['Charged Off', 'Default', 'Late (16-30 days)', 'Late (31-120 days)']), 'loan_status'] = 1
        modified_df['loan_status'] = modified_df['loan_status'].astype('int')
        self.df = modified_df
        
    ## 5. 데이터 처리용 함수
    def __delete_suffix(self, term:str)->int:
        '''첫 단어만을 저장하는 함수'''
        term = term.strip().split()[0]
        return int(term)

    def __delete_suffix_percentage(self, term:str)->float:
        '''%를 자르는 함수'''
        term = term.strip('%')
        return float(term)
    
    def __fill_na_with_value(self, columns:List[str], filling_value:Union[str, int])->None:
        '''
        df: dataframe to fill NA
        column_name : column name to change NA values
        filling_value : value type or just value to fill column's NA
        '''
        for column_name in columns:
            if filling_value=="mode":
                mode_value = self.df[column_name].mode()[0]
            elif filling_value=="median":
                mode_value = self.df[column_name].median()
            else:
                mode_value = filling_value
            self.df[column_name].fillna(mode_value, inplace=True)
        
    def __preprocessing_na(self)->None:
        ## 결측 처리
        # 결측 개수가 1천 건 이하인 경우는 해당 데이터(row) 삭제
        self.df.dropna(subset=['chargeoff_within_12_mths','collections_12_mths_ex_med','dti',
                                                'pub_rec_bankruptcies','revol_util','tax_liens'], inplace=True)
        
        # A. 결측 0 대체
        self.__fill_na_with_value(columns=['annual_inc_joint','dti_joint','revol_bal_joint', 'open_acc_6m',
                                           'open_act_il', 'open_il_12m', 'open_il_24m', 'total_bal_il',
                                           'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'total_cu_tl', 'mths_since_rcnt_il',
                                           'tot_cur_bal', 'total_rev_hi_lim', 'mo_sin_old_rev_tl_op',
                                           'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acc', 'num_bc_sats', 'num_bc_tl',
                                           'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl'], filling_value=0)
        
        # B. 최빈값 대체
        self.__fill_na_with_value(columns=['mo_sin_old_il_acct', 'mths_since_recent_bc', 'mths_since_recent_inq', 'emp_length'], filling_value='mode')
        
        # C. 2015년 대체
        # is_after_2015 컬럼 생성. all_util 변수를 기준으로 사용
        self.df['is_after_2015'] = self.df['all_util'].apply(lambda x: 0 if pd.isnull(x) else 1)
        # 결측값을 0으로 채우기
        
        # D. 2012년 대체
        # is_after_2012 컬럼 생성. pct_tl_nvr_dlq 변수를 기준으로 사용
        self.df['is_after_2012'] = self.df['pct_tl_nvr_dlq'].apply(lambda x: 0 if pd.isnull(x) else 1)
        
    def __convert_object_to_numeric(self, column_name:str)->pd.DataFrame:
        unique_values = sorted(self.df[column_name].unique())
        value_map = {value:index for index, value in enumerate(unique_values)}
        self.df[column_name] = self.df[column_name].apply(lambda x:value_map.get(x))
        return self.df
    
    def __convert_object_to_one_hot(self, column_name:str)->None:
        encoded = pd.get_dummies(self.df[column_name])
        self.df = pd.concat([self.df, encoded], axis=1)
        self.df.drop(column_name, axis=1, inplace=True)
        
    def __preprocessing_objects(self)->None:
        ## TODO : 'addr_state' 필드 해결하기
        # term
        self.df['term'] = self.df['term'].apply(self.__delete_suffix)
        # emp_length
        self.df['emp_length'] = self.df['emp_length'].apply(lambda x: x.replace(' years','').replace(' year','').replace('+','').replace('< 1', '0'))
        self.df['emp_length'] = self.df['emp_length'].astype(int)
        # revol_util
        self.df['revol_util'] = self.df['revol_util'].apply(self.__delete_suffix_percentage)
        
        ## numeric
        # application_type
        self.df = self.__convert_object_to_numeric('application_type')
        # sub_grade
        self.df = self.__convert_object_to_numeric('sub_grade')

        ## one-hot
        # home_ownership
        self.df['home_ownership'] = self.df['home_ownership'].replace(['ANY', 'OTHER', 'NONE'], 'OTHERS')
        self.__convert_object_to_one_hot('home_ownership')
        # purpose
        self.__convert_object_to_one_hot('purpose')
        # verification_status
        self.__convert_object_to_one_hot('verification_status')
        # addr_state : 해야함...

    def preprocess(self)->None:
        # loan_status 제외 모든 column이 결측치(na)인 행 제거 (1개 행 제거됨)
        self.df.dropna(subset=self.df.columns.difference(['loan_status']),how='all', inplace=True)
        self.__preprocess_target_variable()
        # 결측치 제거
        self.__preprocessing_na()
        ## object 처리하기
        self.__preprocessing_objects()
        # index 재설정
        self.df.reset_index(drop=True, inplace=True)
        self.df.dropna(subset=self.df.columns.difference(['loan_status']),how='all', inplace=True)
        print(self.df['application_type'].head(10))    
        
    def get_df(self)->pd.DataFrame:
        return self.df