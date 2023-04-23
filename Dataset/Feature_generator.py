import pandas as pd
from tqdm import tqdm
import pymatgen.core as mg
import numpy as np
#%%
df = pd.read_csv('10387_2_noP.csv', encoding="iso-8859-1")
NAME_HEAs = list (df['Alloy'])

#%% creating an empty dataframe with columns of elements and row of alloys
element_symbols = ['Num_el','H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                   'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
                   'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                   'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
                   'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
                   'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
                   'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
                   'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                   'Tl', 'Pb', 'Bi', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
                   'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg',
                   'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv',
                   'Ts', 'Og']

# Create an empty dataframe with columns named after element symbols
df2 = pd.DataFrame(columns=element_symbols)

for s in tqdm(range (len(NAME_HEAs))):
    df2.loc[s] = [0]*len(element_symbols)
    
    
df2.index = NAME_HEAs

#%% to fill the above data frame
i = 0
for name in tqdm(NAME_HEAs):
    # print(name)
    
    HEAs = mg.Composition(name)
    sym = []
    frac = []
    
    for i in range (len(HEAs)):
        el = HEAs.elements[i].symbol
        el_frac = HEAs.get_atomic_fraction(el)
        sym.append(el)
        frac.append(el_frac)
    
    for j in range (len(sym)):
        df2.loc[name, 'Num_el'] = len(sym)
        df2.loc[name, sym[j]] = frac[j] 
#%%
cols_with_all_zeros = df2.columns[(df2 == 0).all()] #delete columns with zero values
df2 = df2.drop(cols_with_all_zeros, axis=1)
df2 = df2.iloc[:, :-5] #delete columns with none values
#%%
class Read_prepare_data:
    def __init__(self, name='elemental_propertieses2.csv'):
        self.df = pd.read_csv(name)
        # self.df.drop('Unnamed: 0',inplace=True, axis=1)
        # self.df.sort_values(by=['atomic_number'], ascending=True, inplace=True, ignore_index=True)
        self.df.set_index('symbol', inplace=True)
    def list_att(self):
        return self.df.columns.tolist()

    def keep_att(self, list_keep): #When the number of attributes or discroptors is huge we just keep some of them
        self.df = self.df[list_keep]
        return  self.df.dropna()
    
    def drop_att(self, list_drop):
        
        for i in list_drop:
            self.df.drop(i, inplace=True, axis=1)
            
        return self.df
        
#%%
obj = Read_prepare_data()
att = obj.list_att()
my_choice2 = ['VEC', 'Pauling_EN', 'Melting_point_K','DFT_LDA_Etot', 
'outer_shell_electrons', 'no_of_valence_electrons',
              'Atomic_radius_calculated', 'Atomic_weight']
df_element_attribute = obj.keep_att(my_choice2)
#%%
corr_matrix = df_element_attribute.corr()

#%%
class Mixture:
    def __init__(self, name, ATT, df):
        self.HEAs = mg.Composition(name)
        self.sym = []
        self.frac = []
        self.df = df
        self.ATT = ATT
    
        for i in range (len(self.HEAs)):
            self.el1 = self.HEAs.elements[i].symbol
            self.el1_frac = self.HEAs.get_atomic_fraction(self.el1)
            
            self.sym.append(self.el1)
            self.frac.append(self.el1_frac)
            
        self.b = []
        for m in self.sym:
            self.condition = self.df.index == m
            self.indices = self.df.index[self.condition]
            self.var = self.df.loc[self.indices, self.ATT]
            self.var2 = pd.Series.to_numpy(self.var)
            self.var2 = self.var2[0]
            
            self.b.append(self.var2)
         
    def mix(self):
        self.b = np.array(self.b)
        self.frac = np.array(self.frac)
        
        self.final = np.dot(self.b, self.frac)
        return self.final
#%%
att_name = list(df_element_attribute.columns)
#%%
whole = []
i=0
for Name in tqdm(NAME_HEAs):
    print(Name)
    i=i+1
    print(i)
    
    for at in att_name:
        ex = Mixture(Name, at, df_element_attribute)
        tmp = ex.mix()
        whole.append(tmp)
whole = np.reshape(whole, (len(NAME_HEAs), len(att_name)))
data = pd.DataFrame(whole, index = NAME_HEAs, columns=att_name)

#%%
#sqrt(sigma(i-ave)**2)
lst_difs = ['Pauling_EN']

for ii, lst_dif in enumerate(lst_difs):
    ind = list(data.index)
    dif_lst = []
    for iterator in ind:
        
        av = data.loc[iterator, lst_dif]
        if isinstance(av, pd.core.series.Series):
            av = av.iloc[0]
        sum_power_two_mul_frac = 0
        com = mg.Composition(iterator)
        for el in range (len(com)):
            symb = com.elements[el].symbol
            fra  = com.get_atomic_fraction(symb) 
            i_th_att = df_element_attribute.loc[symb, lst_dif]
            power_two = (i_th_att-av)**2
            power_two_mul_frac = fra * power_two
            sum_power_two_mul_frac+= power_two_mul_frac
            
        final = np.sqrt(sum_power_two_mul_frac)
        dif_lst.append(final)
        
    
    data[lst_dif+'_div'] = dif_lst


#%% Entropy
ind = list(data.index)
entropy = 0
entropy_list = []
for iterator in ind:
    com = mg.Composition(iterator)
    entropy = 0
    for el in range (len(com)):
        symb = com.elements[el].symbol
        fra  = com.get_atomic_fraction(symb) 
        entropy += -8.314*fra * ((np.log(fra)))
    entropy_list.append(entropy)


data['entropy'] = entropy_list

#%%
#sqrt(sigma(1-i/ave)**2)
lst_difs = ['Atomic_radius_calculated']

for ii, lst_dif in enumerate(lst_difs):
    ind = list(data.index)
    dif_lst = []
    for iterator in ind:
        
        av = data.loc[iterator, lst_dif]
        if isinstance(av, pd.core.series.Series):
            av = av.iloc[0]
        sum_power_two_mul_frac = 0
        com = mg.Composition(iterator)
        for el in range (len(com)):
            symb = com.elements[el].symbol
            fra  = com.get_atomic_fraction(symb) 
            i_th_att = df_element_attribute.loc[symb, lst_dif]
            power_two = (1.0-i_th_att/av)**2
            power_two_mul_frac = fra * power_two
            sum_power_two_mul_frac+= power_two_mul_frac
            
        final = np.sqrt(sum_power_two_mul_frac)
        dif_lst.append(final)
        
    
    data[lst_dif+'_dif'] = dif_lst


#%% mixing enthalpy 4Hcicj
ind = list(data.index)

df_enthalpy = pd.read_csv('Enthalpy_mix.csv')
df_enthalpy.set_index('symbol', inplace=True)
df_enthalpy.drop('H',inplace=True, axis=1)

mix_enthalpy = []
for iterator in ind:
    print(iterator)
    com = mg.Composition(iterator)

    symb = []
    frac = []
    for i in range (len(com)):
        sym = com.elements[i].symbol
        # el1_frac = com.get_atomic_fraction(sym)
        symb.append(sym)
        # frac.append(el1_frac)
    agg = 0

    for k in symb:
        for n in symb:
            if k==n: 
                continue
            else:
                mix = df_enthalpy.loc[k,n]
                k_frac = com.get_atomic_fraction(k)
                n_frac = com.get_atomic_fraction(n)
                agg += mix * k_frac * n_frac
                
    mix_enthalpy.append(2*agg)



data['Enthalpy'] = mix_enthalpy
#%%

df_concatenated = pd.concat([df2, data], axis=1)
#%%

y = df["Phase"].copy()
y = y.to_frame()
data_rest = df_concatenated.reset_index()
data_rest['Geo'] = data_rest['entropy'] / (data_rest['Atomic_radius_calculated_dif'])**2
data_final = pd.concat([data_rest, y], axis=1)
data_final['Atomic_radius_calculated_dif'] = data_final['Atomic_radius_calculated_dif']*100
data_final['E_per_el'] = data_final['DFT_LDA_Etot']/data_final['Num_el']
# data_final.to_csv('dataset10387_70.csv')
