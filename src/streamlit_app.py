import pandas as pd
import streamlit as st
from custom_functions import transform_data, modelos_ML

# Definir las variables data_2018, data_2019 y data_2020 antes de llamar a la función transform_data
data_2018 = pd.read_csv('./data/ground_water_quality_2018_post.csv')
data_2019 = pd.read_csv('./data/ground_water_quality_2019_post.csv')
data_2020 = pd.read_csv('./data/ground_water_quality_2020_post.csv')  # Utilizaremos el 2020 para el test

# Llama a la función transform_data para transformar tus datos
data_2018, data_2019, data_2020, data_agua = transform_data(data_2018, data_2019, data_2020)

resultados = modelos_ML(data_agua, data_2020)

# Definir la interfaz de la aplicación
st.title('Aplicación de Predicción de Calidad del Agua')
st.write('Ingresa los valores de las características para obtener una predicción.')
st.write('Cono De Paola')

# Definimos las opciones para columnas categóricas

district_options = ['ADILABAD', 'BHADRADRI', 'BHUPALPALLY', 'HYDERABAD', 'JAGITYAL',
       'JANGAON', 'JOGULAMBA(GADWAL)', 'KAMAREDDY', 'KARIMNAGAR',
       'KHAMMAM', 'KUMURAM BHEEM', 'MAHABUBABAD', 'MAHABUBNAGAR',
       'MANCHERIAL', 'MEDAK', 'MEDCHAL', 'MULUGU', 'NAGARKURNOOL',
       'NALGONDA', 'NARAYANPET', 'NIRMAL', 'NIZAMABAD', 'PEDDAPALLY',
       'RANGAREDDY', 'SANGAREDDY', 'SIDDIPET', 'SIRCILLA', 'SURYAPET',
       'VIKARABAD', 'WANAPARTHY', 'WARANGAL (R) ', 'WARANGAL (U)',
       'YADADRI']

# mandal_options = ['Adilabad', 'Bazarhatnur', 'Gudihatnoor', 'Jainath', 'Narnoor',
#        'Neradigonda', 'Talamadugu', 'Tamsi', 'Utnoor', 'Annapureddypalli',
#        'Ashwapuram', 'Bhadrachalam', 'Burgampad', 'Chandrugonda',
#        'Dummugudem', 'Gundala', 'Julurpadu', 'Karakagudem',
#        'Laxmidevipalli', 'Manuguru', 'Palwancha', 'Tekulapalli',
#        'Yellandu', 'Ghanpur Mulug', 'Kataram', 'Mahadevpur', 'Regonda',
#        'Ameerpet', 'Asifnagar', 'Bandlaguda', 'Charminar', 'Maredpally',
#        'Nampally', 'Saidabad', 'Dharmapuri', 'Gollapalli',
#        'Ibrahimpatnam', 'Jagityal', 'Kathalapur', 'Kodimal', 'Korutla',
#        'Mallapur', 'Mallial', 'Metpalli', 'Pegadapalli', 'Raikal',
#        'Sarangapur', 'Velgatur', 'Bachannapet', 'Devaruppula', 'Jangaon',
#        'Lingala ghanpur', 'Palakurthi', 'Raghunadhapalli', 'Zafergad',
#        'Alampur', 'Dharoor', 'Gadwal', 'Gattu', 'Ieeza', 'K-T Doddi',
#        'Maldakal', 'Manopad', 'Waddepalli', 'Bansawada', 'Bhiknoor',
#        'Bibipet', 'BICHKUNDA', 'Domakonda', 'Gandhari', 'Jukkal',
#        'Kamareddy', 'Lingampet', 'Machareddy', 'Maddnur', 'Nagireddipet',
#        'Nasurullabad', 'Nizamsagar', 'Rajampet', 'RAMAREDDY',
#        'Sadasivanagar', 'Tadwai', 'Choppadandi', 'Gangadhara',
#        'Huzurabad', 'Saidapur', 'Thimmapur', 'Bonakal', 'Kallur',
#        'Kamepalli', 'Khammam (R )', 'Khammam(U)', 'Konijerla',
#        'Kusumanchi', 'Mudigonda', 'Sathupalli', 'Thallada',
#        'Thirumalayapalem', 'Vemsur', 'Yerrupalem', 'Asifabad',
#        'Kagaznagar', 'Kerameri', 'Kowthala', 'Rebbena', 'Tiryani',
#        'Dhantalapally', 'Dornakal', 'Kesamudram', 'Koravi', 'Mahabubabad',
#        'Marripeda', 'Nellikudur', 'Torrur', 'Bhoothpur', 'CC Kunta',
#        'Devarkadara', 'Hanwada', 'Jadcherla', 'Koilkonda',
#        'Mahabubnagar (R) ', 'Mahabubnagar(U)', 'Midjil', 'Nawabpet',
#        'Bellampally', 'Chennur', 'Jaipur', 'Mandamarri', 'Tandur',
#        'Vemanpally', 'Chegunta', 'Havelighanpur', 'Kulcharam', 'Medak',
#        'Narasapur', 'Papannapet', 'Regode', 'Shankarampet ',
#        'Shankarampet R', 'Shivampet', 'Tekmal', 'Toopran', 'Yeldurthy',
#        'Alwal', 'Balanagar', 'Kukatpally', 'Malkajgiri', 'Qutubullapur',
#        'Eturu Nagaram', 'Govindaraopet', 'Amarabad', 'Balmur',
#        'Bijinepalli', 'Kalwakuthy', 'Kodair', 'Kollapur', 'Lingal',
#        'Nagarkurnool', 'Telkapally', 'Thimmajipet', 'Uppununuthala',
#        'Veldanda', 'Advidevulapally', 'Anumula', 'Chandampet', 'Chandur',
#        'Chityala', 'Devarakonda', 'Gundlapally', 'Gurrampode',
#        'K.Mallepally', 'kanagala', 'Kattangur', 'Marriguda', 'Munugode',
#        'Nakrekal', 'Nalgonda', 'Narketpalli', 'Nidamanuru', 'P.A Pally',
#        'Peddavoora', 'Shaligowraram', 'Thiparthy', 'Kosigi', 'Maddur',
#        'Makthal', 'Marikal', 'Narayanpet', 'Narva', 'Utkoor', 'Kadam',
#        'Khanapur', 'Kubeer', 'Kuntala', 'Laxmanchanda', 'Lokeswaram',
#        'Mamada', 'Nirmal', 'Tanur', 'Armoor', 'Bheemgal', 'Bodhan',
#        'Darpally', 'Dichpally', 'Indalwai', 'Jakrampally', 'Kammarapalli',
#        'Kotagiri', 'Morthad', 'Mugpal', 'Mupkal', 'Nandipet', 'Nizamabad',
#        'Renjal', 'Rudrur', 'Sirikonda', 'Vailpoor', 'Varni', 'Yedapalli',
#        'Dharmaram', 'Eligedu', 'Julapalli', 'Odella', 'Peddapalli',
#        'Chevella', 'Amangal', 'Serilingampally', 'Kandukur', 'Keshampet',
#        'Kothur', 'Madgula', 'Maheswaram', 'Manchal', 'Moinabad',
#        'Shamshabad', 'Rajender Nagar', 'Farooqnagar', 'Shahbad',
#        'Shankarpalli', 'Talakondapalli', 'Yacharam', 'Andole',
#        'Gummadiddla', 'Hathnoora', 'Jinnaram', 'Kalher', 'Kandi',
#        'Patancheru ', 'R.C.Puram', 'Gajwel', 'Jagdevpur', 'Mirdoddi',
#        'Mulugu', 'Nanganur', 'Raipole', 'Siddipet', 'Wargal',
#        'Gambhiraopet', 'Illanthakunta', 'Konaraopet', 'Sircilla',
#        'Vemulawada', 'Yellareddipet', 'Arvapally', 'Chivemla',
#        'Mattampalle', 'Mellacheruvu', 'Nuthankal', 'Thungathurthi',
#        'Bantwaram', 'Basheerabad', 'Bomraspet', 'Doma', 'Doulathabad',
#        'Kodangal', 'Marpalli', 'Parigi', 'Peddamul', 'Pudur',
#        'Vikarabad ', 'Yalal', 'Atmakuru', 'Ghanpur', 'Gopalpet',
#        'Kothakota', 'Pangal', 'Pebbair', 'Peddamandadi', 'Atmakur',
#        'Chennaraopet', 'Duggondi', 'Narasampet', 'Nekkonda', 'Parkal',
#        'Parvathagiri', 'Rayaparthy', 'Sangem', 'Wardhannapet',
#        'Bheemadevarapallly', 'Dharmasagar', 'Hanamkonda', 'Hasanparthi',
#        'Kamalapur', 'warangal', 'Alair', 'B.Pochampalli', 'Choutuppal',
#        'Mothukur', 'Rajapet', 'Ramannapet', 'S.Narayanpur', 'Thurkapally',
#        'Valigonda', 'Y.Gutta', 'Bibinagar']

# village_options = ['Adilabad', 'Bazarhatnur', 'Gudihatnoor', 'Jainath', 'Narnoor',
#        'Neradigonda', 'Talamadugu', 'Tamsi', 'Utnoor', 'Annapureddypalli',
#        'Aswapuram', 'Bhadrachalam', 'M.Banjara', 'Ravikampadu',
#        'Bandirevu', 'Kanchanapalli', 'P. Narsapuram', 'Karakagudem',
#        'Laxmidevipalli', 'Pagideru', 'REGELLA', 'Bethampudi',
#        'Mutyalampadu', 'Komararam', 'Chelpur(D)', 'Shankarampally 125(D)',
#        'Ambatipalli', 'Regonda', 'S.R.Nagar', 'Himayanagar', 'Kulsanapur',
#        'Chandraingutta', 'Darul Shifa', 'Maredpally(s)', 'Nampally',
#        'Juvinile home', 'Dharmapuri', 'Gollapalli', 'Ibrahimpatnam',
#        'Jagityal', 'Kathalapur', 'Chepial', 'Korutla', 'Mallapur',
#        'Nukapalli', 'Metpalli', 'Pegadapalli', 'Raikal', 'Sarangapur',
#        'Velgatur', 'Bachannapet', 'Singarajupalli', 'Jangaon',
#        'Lingala ghanpur', 'Palakurthi', 'Raghunadhapalli', 'Zafergad',
#        'Alampur', 'Dharoor', 'Neelahally', 'Gadwal', 'Gattu', 'Ieeza',
#        'Kondapur', 'Kuchinerla', 'Maldakal', 'Manopad', 'Santhinagar',
#        'Bansawada', 'Bhiknoor', 'P.D.Mallareddy', 'Bibipet', 'Pulkal',
#        'Domakonda', 'Gandhari', 'Sarvapoor', 'Jukkal', 'Adloor',
#        'Narasannapally', 'Bhavanipet', 'Machareddy', 'Menoor',
#        'Malthummeda', 'Nasurullabad', 'Mohammadnagar', 'Argonda',
#        'Reddypet', 'Sadasivanagar', 'Yerrapahad', 'Rukmapur',
#        'Gangadhara', 'Huzurabad', 'Saidapur', 'Alugunur', 'Mustikunta',
#        'Kallur', 'Kothalingala', 'M.V.Palem', 'Khammam I', 'Khammam-II',
#        'Konijerla', 'Kusumanchi', 'Mudigonda', 'Prakashnagar',
#        'Anjanapuram', 'Bachode', 'Tirumalayapalem', 'Vemsur',
#        'Banigandlapadu', 'Asifabad', 'Kagaznagar', 'Kerameri', 'Kowthala',
#        'Rebbena II', 'Tiryani', 'Dhantalapally', 'Mannegudem',
#        'Kesamudram', 'Ayyagaripally', 'Mahabubabad', 'Marripeda',
#        'Nellikudur', 'Torrur', 'Elkicherla', 'Damagnapur', 'Kurumurthy',
#        'Thirmalapur', 'Devarakadra', 'Hanwada', 'Jadcherla',
#        'Dammaipally', 'Kodur', 'Yenugonda', 'Kothapalli', 'Nawabpet',
#        'Bellampally I', 'Chennur I', 'Jaipur', 'Mandamarri', 'Tandur ',
#        'Neelwai', 'Chegunta', 'Mutayapally', 'Rangampet', 'Medak',
#        'Narasapur', 'Reddypally', 'Papannapet', 'T.lingampally',
#        'Gavvalapalli', 'Shankarampet', 'Shivampet', 'Gudur',
#        'Usrikapally', 'Tekmal', 'Islampur', 'Toopran', 'Edulapally',
#        'Kukunoor', 'Ramayapally', 'Old Alwal', 'Balanagar 1',
#        'Kaithalapur', 'Kukatpally 1', 'Malkajgiri', 'Gajularamaram 2',
#        'Qutubullapur 2', 'Eturu Nagaram (D)', 'Eturu Nagaram (S)',
#        'Pasra', 'Project Nagar', 'Domalapenta', 'Balmur', 'Bijinepalli',
#        'Kalwakuthy', 'Kodair', 'Kollapur', 'Ambatipally', 'Nagarkurnool',
#        'Telkapally', 'Gummakonda', 'Uppununthala', 'Veldanda',
#        'Ulsaipalem', 'Anumula', 'Chintagudem', 'Rajavaram', 'Sreerampur',
#        'Chandampet', 'Angadipet', 'Bangarigadda', 'Chandur', 'Sirdepally',
#        'Veliminedu', 'Padamtipally', 'Tatikole', 'Vavikole', 'Gurrampode',
#        'Koppole', 'K.Mallepally', 'kanagala', 'Cheruvu Annaram',
#        'Kattangur', 'Marriguda', 'Kistapur', 'Kompalli', 'Arlagaddagudam',
#        'Nakrekal', 'Vallabhapur', 'Mushampally', 'S L B C G V guda',
#        'Seetarampuram', 'Akkenepally', 'Narketpalli', 'Nidamanur',
#        'Velmaguda', 'P.Kondaram', 'Thipparthy', 'Mamidala', 'Kosigi',
#        'Maddur', 'Makthal', 'Marikal', 'Kollampally', 'Kotakonda',
#        'Narayanapet', 'Narva', 'Utkoor', 'Chinnaporla', 'Kadam',
#        'Khanapur', 'Kubeer', 'Kuntala', 'Laxmanchanda', 'Abdullapur',
#        'Lokeswaram', 'Mamada', 'Nirmal', 'Beeravelly', 'Bhosi', 'perkit',
#        'Bheemgal', 'Bodhan', 'Ramadugu', 'Yanampally', 'Gannaram',
#        'Jakrampally', 'Inayathnagar', 'Morthad', 'Manchippa', 'Mupkal',
#        'Ailapoor', 'Nutpally', 'Velmal', 'Arsapally', 'Dupally',
#        'Rayakur', 'Peddavolgote', 'Ankushpur', 'Vailpoor', 'Varni',
#        'Yedapalli', 'Dharmaram', 'Eligedu', 'Julapalli', 'Odella',
#        'Peddapalli', 'Alur', 'Amangal', 'Dharmasagar', 'Gachibowli',
#        'Kandukur', 'Keshampet', 'Kothur', 'Madgula', 'Maheswaram',
#        'Manchal', 'Mangalpally', 'Moinabad', 'Palmokole',
#        'Rajender Nagar', 'Shadnagar', 'Shahbad', 'Shankarpalli',
#        'Talakondapalli', 'Yacharam', 'Jogipet', 'Gummadiddla',
#        'Borapatla', 'Nasthipur', 'Ootla', 'Bachapally', 'Meerkhanpally',
#        'Byathole', 'Isnapur', 'Patancheruvu', 'R.C.Puram', 'Gajwel',
#        'Munigadapa', 'Bhoompalli', 'Mulugu', 'Rampur', 'Raipole',
#        'Siddipet', 'Majidpalli', 'Narmal', 'Illanthukunta ', 'Nizamabad',
#        'Sircilla(urban)', 'Vemulawada(rural)', 'Racherla Boppaspur',
#        'Nagaram', 'B. Chandupatla', 'Raghunadhapalem', 'Mellacheruvu',
#        'Nuthankal', 'Thungathurthi', 'Bantwaram', 'Basheerabad',
#        'Bomraspet', 'Doma', 'Doulathabad', 'Kodangal', 'Rudraram (S)',
#        'Marpalli', 'Vattinenipally(s)', 'Parigi', 'Peddamul', 'Pudur',
#        'Tandur', 'Vikarabad(S)', 'Yalal', 'Atmakuru', 'Ghanapur',
#        'Gopalpet', 'Kanaipally', 'Pangal', 'Ayyavaripalli',
#        'Peddamandadi', 'Atmakur', 'Chennaraopet', 'Duggondi',
#        'Narasampet', 'Ameenpet', 'Parkal', 'Parvathagiri', 'Rayaparthy',
#        'Sangem', 'Wardhannapet', 'Vangara', 'Narayanagiri', 'Hanumkonda',
#        'Mamnoor', 'Seethampet', 'Shanigaram', 'Charbowli', 'Kolanpaka',
#        'B.Pochampalli', 'D.Malkapur', 'Bondugala', 'Rajapet', 'Somaram',
#        'Ramanapet', 'S.Narayanpur', 'Gandamalla', 'T. somaram',
#        'Vemulakonda', 'Mallapuram', 'Shapurnagar', 'Bibinagar',
#        'Kurraram']




# Creamos las listas desplegables para columnas categóricas
district = st.selectbox('District', district_options)
# mandal = st.selectbox('Mandal', mandal_options)
# village = st.selectbox('Village', village_options)

# Valores medios de las características
mean_values = {
    'lat_gis' : 17.708454,
    'long_gis' : 78.788021,
    'gwl': 10.242452,
    'ph': 7.854038,
    'ec': 1336.079946,
    'tds': 855.091165,
    'co3': 11.287418,
    'hco3': 295.692665,
    'cl': 188.590786,
    'f': 1.166641,
    'no3': 72.358543,
    'so4': 46.212398,
    'na': 123.504072,
    'k': 8.069593,
    'ca': 80.836314,
    'mg': 50.807989,
    'th': 410.868109,
    'sar': 2.795812,
    'RSC meq / L' : -2.182887
}

# Crear campos de entrada con valores iniciales basados en medias

lat = st.number_input('Latitud (lat_gis)', min_value=0.0, value=mean_values['lat_gis'], step=1.0)
long = st.number_input('Longitud (long_gis)', min_value=0.0, value=mean_values['long_gis'], step=1.0)
gwl = st.number_input('Nivel de Agua Subterránea (gwl)', min_value=0.0, value=mean_values['gwl'], step=1.0)
ph = st.number_input('pH', min_value=0.0, value=mean_values['ph'], step=1.0)
ec = st.number_input('Conductividad Eléctrica (E.C)', min_value=0.0, value=mean_values['ec'], step=1.0)
tds = st.number_input('Total de Sólidos Disueltos (TDS)', min_value=0.0, value=mean_values['tds'], step=1.0)
co3 = st.number_input('Carbonatos (CO3)', min_value=0.0, value=mean_values['co3'], step=1.00)
hco3 = st.number_input('Bicarbonatos (HCO3)', min_value=0.0, value=mean_values['hco3'], step=1.0)
cl = st.number_input('Cloro (Cl)', min_value=0.0, value=mean_values['cl'], step=1.0)
f = st.number_input('Fluoruro (F)', min_value=0.0, value=mean_values['f'], step=1.00)
no3 = st.number_input('Nitrato (NO3)', min_value=0.0, value=mean_values['no3'], step=1.0)
so4 = st.number_input('Sulfato (SO4)', min_value=0.0, value=mean_values['so4'], step=1.0)
na = st.number_input('Sodio (Na)', min_value=0.0, value=mean_values['na'], step=1.0)
k = st.number_input('Potasio (K)', min_value=0.0, value=mean_values['k'], step=1.0)
ca = st.number_input('Calcio (Ca)', min_value=0.0, value=mean_values['ca'], step=1.0)
mg = st.number_input('Magnesio (Mg)', min_value=0.0, value=mean_values['mg'], step=1.0)
th = st.number_input('Dureza Total (T.H)', min_value=0.0, value=mean_values['th'], step=1.0)
sar = st.number_input('Relación de Adsorción de Sodio (SAR)', min_value=0.0, value=mean_values['sar'], step=1.0)
rsc1 = st.number_input('Residuos sólidos disueltos (RSC meq)', min_value=-10.0,value=mean_values['RSC meq / L'], step=1.0)

# Cuando el usuario haga clic en un botón "Predecir", obtén los valores ingresados
if st.button('Predecir'):
    # Crea un DataFrame con los valores ingresados por el usuario
    input_data = pd.DataFrame({
        'lat_gis': [lat],
        'long_gis': [long],
        'gwl': [gwl],
        'ph': [ph],
        'ec': [ec],
        'tds': [tds],
        'co3': [co3],
        'hco3': [hco3],
        'cl': [cl],
        'f': [f],
        'no3': [no3],
        'so4': [so4],
        'na': [na],
        'k': [k],
        'ca': [ca],
        'mg': [mg],
        'th': [th],
        'sar': [sar],
        'RSC': [rsc1]
    })


# Utiliza el modelo seleccionado para hacer la predicción
modelo_seleccionado = st.selectbox('Selecciona un modelo:', ('Árbol de Decisión', 'Bosque Aleatorio', 'Máquina de Vectores de Soporte (SVM)'))

# # Botón para generar resultados con los valores ingresados
# if st.button('Generar Resultados'):
#     # Aquí puedes realizar cualquier cálculo o procesamiento necesario con los valores ingresados
#     # Por ejemplo, puedes imprimir los valores seleccionados y numéricos
#     st.write(f'District: {district}')
#     st.write(f'Mandal: {mandal}')
#     st.write(f'Village: {village}')
#     st.write(f'Nivel de Agua Subterránea (gwl): {gwl}')
#     st.write(f'pH: {ph}')
#     st.write(f'Conductividad Eléctrica (E.C): {ec}')
#     st.write(f'Total de Sólidos Disueltos (TDS): {tds}')
#     st.write(f'Carbonatos (CO3): {co3}')
#     st.write(f'Bicarbonatos (HCO3): {hco3}')
#     st.write(f'Cloro (Cl): {cl}')
#     st.write(f'Fluoruro (F): {f}')
#     st.write(f'Nitrato (NO3): {no3}')
#     st.write(f'Sulfato (SO4): {so4}')
#     st.write(f'Sodio (Na): {na}')
#     st.write(f'Potasio (K): {k}')
#     st.write(f'Calcio (Ca): {ca}')
#     st.write(f'Magnesio (Mg): {mg}')
#     st.write(f'Dureza Total (T.H): {th}')
#     st.write(f'Relación de Adsorción de Sodio (SAR): {sar}')

if modelo_seleccionado == 'Árbol de Decisión':
        modelo = resultados['dt_classifier']
elif modelo_seleccionado == 'Bosque Aleatorio':
        modelo = resultados['rf_classifier']
else:
        modelo = resultados['svm_classifier']

predicciones = modelo.predict(input_data)

st.write(f'La predicción con el modelo {modelo_seleccionado} es: {predicciones[0]}')
