import numpy as np 
import pandas as pd
import xml.etree.ElementTree as ET

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline 

class RealestateTypeTransformer( BaseEstimator, TransformerMixin ):
    
    def fit( self, X, y = None ):
        return self
    
    def transform( self, X, y = None ):
        self.X = X
        X['action'] = X['Tüüp'].map(lambda t : self.extractAction(t))
        X['estate_type'] = X['Tüüp'].map(lambda t : self.extractRealestateType(t))
        return X

    def extractAction(self, row_type):
        action_type = ''
        row_type = row_type.lower()
        if 'müüa' in row_type:
            action_type += 'SALE'
        elif 'anda üürile' in row_type:
            action_type += 'RENT'
        else:
            return 'UNKNOWN'
        
    #    if 'vahetuse võim' in row_type:
    #        action_type += '_AND_EXCHANGE'
            
        return action_type

    def extractRealestateType(self, row_type):
        realestate_type = ''
        row_type = row_type.lower()
        if 'korter' in row_type:
            action_type = 'APT'
        elif 'majaosa' in row_type:
            action_type = 'PART_OF_HOUSE'
        elif 'äripind' in row_type:
            action_type = 'BUSINESS'
        elif 'maja' in row_type:
            action_type = 'HOUSE'
        elif 'ridaelamuboks' in row_type:
            action_type = 'TERR_HOUSE'
        elif 'suvila' in row_type:
            action_type = 'COTT_HOUSE'
        elif 'talu' in row_type:
            action_type = 'FARM'
        elif 'garaaž' in row_type:
            action_type = 'GARAGE'
        else:
            action_type = 'UNKNOWN'
            
        return action_type

class NoTransformer( BaseEstimator, TransformerMixin ):
        
    def fit( self, X, y = None ):
        self.X = X
        return self
    
    def transform( self, X, y = None ):
        return X

class CountyTransformer( BaseEstimator, TransformerMixin ):

    administrative_entities = ['küla', 'alevik', 'vald', 'linn', 'maakond', 'linnaosa', 'alev']

    def fit( self, X, y = None ):
        self.admin_div = self.getCountyDivs()
        return self
    
    def transform( self, X, y = None ):
        self.X = X
        X['Maakond'] = X['Maakond'].map(lambda county: self.find_matching_county(county))            
        return X
    
    def find_matching_county( self, input_county ):
        county_mods = self.getCountyMods(input_county)
        for county in self.admin_div:
            is_matching = any(mod in self.admin_div[county] for mod in county_mods)
            if is_matching:
                return county
            
        return 'unknown'

    def getCountyDivs(self):
        tree = ET.parse('data/EHAK2019v8.xml')
        root = tree.getroot()
        county_div_list = {}
        for el in root.findall('./Classification/Item'):
            label = el.find('Label/LabelText').text.lower()
            county_subdivs = self.getCountyMods(label)
            self.parseChildrenToList(el, county_subdivs)
            county_div_list[label] = county_subdivs
        return county_div_list

    def parseChildrenToList(self, root_item, subdivision_list):
        child_items = root_item.findall('Item')
        label = root_item.find('Label/LabelText').text.lower()
        if len(child_items) == 0:
            return self.getCountyMods(label)
        
        for child in child_items:
            subdivision_list.extend(self.parseChildrenToList(child, subdivision_list))
            
        return self.getCountyMods(label)
    

    def getCountyMods(self, original):
        county_mods = []
        original = original.lower()
        county_mods.append(original.lower())
        county_split_ws = original.split()
        county_split_slash = original.split('/')
        county_split_dash = original.split('-')
        
        if original.endswith('maakond'):
            county_mods.append(original.replace(' maakond', 'maa'))
            
        if county_split_ws[-1] in self.administrative_entities:
            county_mods.append(' '.join(county_split_ws[:-1]))
        
        if len(county_split_slash) > 1:
            county_mods.extend(county_split_slash)
        
        if len(county_split_dash) > 1:
            county_mods.extend(county_split_dash)
            
        return county_mods

class PriceOutlierRemoval( BaseEstimator, TransformerMixin ):

    def __init__(self, allowed_z_score=3):
        self.allowed_z_score = allowed_z_score

    def fit( self, X, y = None ):
        return self
    
    def transform( self, X, y = None ):
        X['price_action_z'] =abs((X['Hind'] - X.groupby(['action', 'estate_type'])['Hind'].transform('mean'))\
                                       / X.groupby(['action', 'estate_type'])['Hind'].transform('std'))
        X = X[X['price_action_z'] < self.allowed_z_score]        
        X = X.drop(columns=['price_action_z'])
        return X

class PriceMedianMultiplierExtractor( BaseEstimator, TransformerMixin ):

    def fit( self, X, y = None ):
        self.action_type_price_median = X.groupby(['action', 'estate_type'])['Hind'].median()
        return self
    
    def transform( self, X, y = None ):
        print(self.action_type_price_median)
        X['price_median_multiplier'] = X['Hind'] / X.apply(lambda x : self.action_type_price_median[x['action']][x['estate_type']], axis=1)
        return X