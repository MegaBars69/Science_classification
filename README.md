# Science_classification

# Building a Docker container

    docker build -t serg/tech-transfer .

    docker run -p 8882:5000 serg/tech-transfer

# Begin:

Make a clone to your device:
#
    git clone git@github.com:MegaBars69/Science_classification.git    

To begin using this amazing classifier download requirements:
#    
    pip install -r requirements.txt

After import main file to your program:
#
    from text_classifier import *

# Usefull functions:
classify(text) - classifies text.
#
    classify(text) - returns string
NameToCategories(name, Small = False) - classifies author's works. Second parametr is used if you want to use 'next-cursor'(more works). Creates 3 files 'titles.txt' , 'indexes.txt' (ISSN indexes) and 'output.txt'
#
    NameToCategories(name, Small = False) - returns void
NameToFile(name,Small) - creates 2 files 'titles.txt' and  'indexes.txt' (ISSN indexes) 
#
    NameToFile(name,Small) - returns author rank
ListToCategory(list) - classifies list of strings. create 1 file 'output.txt'
#
    ListToCategory(list) - returns void  

# Examples:
    IN: NameToCategories('Sergey+Afonin',True)
    OUTPUT: 
    Computer Science(Computer Networks and Communications) – Ontology Models for Access Control Systems

    Computer Science(Information Systems) – The View Selection Problem for Regular Path Queries

    Engineering(Electrical and Electronic Engineering) – Characteristics of Nanopositioning Electroelastic Digital-to-Analog Converter for Communication Systems
    ....

    IN: print(classify('Oropharyngeal Stick Injury in a Bengal Cat'))
    OUTPUT: Veterinary(Small Animals)

# Utils:
if someone wants to make you own dataset, Utils folder is for you. 
Firstly, use 'categories.py' to create dataset? by collecting all Areas you need.
After you will need to concat them all to 'csv' file, for this use 'files.py' Normalaze dataset.
Model by creating docs use 'train_model.py' and  train model

# Science Tree:
    Medicine       
    ├── Cardiology and Cardiovascular Medicine                  
    ├── Oncology                                                 
    ├── Neurology (clinical)                                     
    ├── Nephrology                                               
    ├── Pulmonary and Respiratory Medicine                       
    ├── Critical Care and Intensive Care Medicine                
    ├── Hematology                                               
    ├── Internal Medicine                                        
    ├── Surgery                                                  
    ├── Epidemiology                                             
    ├── Pediatrics  Perinatology and Child Health                
    ├── Public Health  Environmental and Occupational Health     
    ├── Psychiatry and Mental Health                             
    ├── Urology                                                  
    ├── Rheumatology                                             
    ├── Hepatology                                               
    ├── Infectious Diseases                                      
    ├── Obstetrics and Gynecology                                
    ├── Radiology  Nuclear Medicine and Imaging                  
    ├── Ophthalmology                                            
    ├── Gastroenterology                                         
    ├── Pathology and Forensic Medicine                           
    └── Molecular Medicine                                
    Social_Sciences/
    ├── Political Science and International Relations    
    ├── Sociology and Political Science                  
    ├── Education                                        
    ├── Geography  Planning and Development              
    ├── Law                                              
    ├── Library and Information Sciences                 
    ├── Communication                                    
    ├── Development                                       
    ├── Demography                                        
    ├── Anthropology                                      
    ├── Cultural Studies                                  
    ├── Gender Studies                                    
    ├── Public Administration                             
    ├── Transportation                                    
    ├── Health (social science)                           
    ├── Transportation                                    
    ├── Linguistics and Language                          
    ├── Life-span and Life-course Studies                 
    ├── Urban Studies                                     
    ├── Human Factors and Ergonomics                      
    └── Archeology                                
    Earth_and_Planetary_Sciences/
    ├── Geology                                             
    ├── Atmospheric Science                                 
    ├── Earth-Surface Processes                              
    ├── Geotechnical Engineering and Engineering Geology     
    ├── Geochemistry and Petrology                           
    ├── Oceanography                                         
    └── Geophysics                     
    Chemistry/
    ├── Organic Chemistry                     
    ├── Physical and Theoretical Chemistry    
    ├── Analytical Chemistry                   
    ├── Inorganic Chemistry                    
    ├── Electrochemistry                       
    ├── Spectroscopy                           
    ├── Environmental Chemistry                
    └── Materials Chemistry                      
    Environmental_Science/
    ├── Water Science and Technology              
    ├── Environmental Chemistry                    
    ├── Waste Management and Disposal              
    ├── Ecology                                    
    ├── Global and Planetary Change                
    ├── Pollution                                  
    ├── Management  Monitoring  Policy and Law      
    ├── Environmental Engineering                   
    └── Nature and Landscape Conservation                          
    Energy/
    ├── Renewable Energy  Sustainability and the Environment    
    ├── Nuclear and High Energy Physics                         
    ├── Nuclear Energy and Engineering                          
    └── Energy Engineering and Power Technology                                            
    Physics_and_Astronomy /
    ├── Condensed Matter Physics                    
    ├── Nuclear and High Energy Physics             
    ├── Astronomy and Astrophysics                  
    ├── Atomic and Molecular Physics  and Optics    
    ├── Instrumentation                             
    ├── Acoustics and Ultrasonics                    
    └── Radiation                              
    Materials_Science/
    ├── Electronic  Optical and Magnetic Materials    
    ├── Ceramics and Composites                        
    ├── Metals and Alloys                              
    ├── Polymers and Plastics                          
    ├── Surfaces  Coatings and Films                    
    ├── Biomaterials                                    
    └── Materials Chemistry                                 
    Neuroscience/
    ├── Behavioral Neuroscience                
    ├── Cellular and Molecular Neuroscience    
    ├── Biological Psychiatry                  
    ├── Neurology                              
    ├── Sensory Systems                        
    ├── Cognitive Neuroscience                 
    ├── Endocrine and Autonomic Systems        
    └── Developmental Neuroscience                                    
    Agricultural_and_Biological_Sciences/
    ├── Food Science                                    
    ├── Plant Science                                    
    ├── Soil Science                                     
    ├── Ecology  Evolution  Behavior and Systematics     
    ├── Insect Science                                   
    └── Aquatic Science              
    Biochemistry_Genetics_and_Molecular_Biology/
    ├── Cell Biology          
    ├── Genetics              
    ├── Molecular Biology     
    ├── Cancer Research       
    ├── Physiology            
    ├── Biochemistry          
    ├── Endocrinology         
    ├── Structural Biology    
    ├── Biophysics             
    └── Molecular Medicine     
    Computer_Science/
    ├── Computer Networks and Communications           
    ├── Software                                       
    ├── Artificial Intelligence                        
    ├── Information Systems                            
    ├── Computer Graphics and Computer-Aided Design     
    ├── Human-Computer Interaction                      
    ├── Hardware and Architecture                        
    ├── Computer Science Applications                    
    ├── Computer Vision and Pattern Recognition          
    ├── Signal Processing                                
    └── Theoretical Computer Science                                
    Business_Management_and_Accounting/
    ├── Marketing                                                
    ├── Strategy and Management                                   
    ├── Tourism  Leisure and Hospitality Management               
    ├── Business and International Management                     
    ├── Accounting                                                
    ├── Management of Technology and Innovation                   
    └── Organizational Behavior and Human Resource Management              
    Psychology/
    ├── Applied Psychology                              
    ├── Developmental and Educational Psychology        
    ├── Clinical Psychology                             
    ├── Neuropsychology and Physiological Psychology    
    ├── Social Psychology                               
    └── Experimental and Cognitive Psychology                                       
    Engineering  /
    ├── Nuclear Energy and Engineering                      
    ├── Electrical and Electronic Engineering                
    ├── Geotechnical Engineering and Engineering Geology     
    ├── Energy Engineering and Power Technology              
    ├── Automotive Engineering                               
    ├── Environmental Engineering                             
    └── Biomedical Engineering                                
    Pharmacology_Toxicology_and_Pharmaceutics/
    ├── Pharmaceutical Science    
    ├── Toxicology                 
    ├── Pharmacology               
    └── Drug Discovery                
    Dentistry/
    ├── Orthodontics    
    ├── Oral Surgery    
    └── Periodontics      
    Mathematics  /
    ├── Applied Mathematics                       
    ├── Analysis                                  
    ├── Algebra and Number Theory                 
    ├── Geometry and Topology                      
    ├── Logic                                      
    ├── Statistics and Probability                 
    ├── Modeling and Simulation                    
    ├── Computational Mathematics                  
    ├── Discrete Mathematics and Combinatorics     
    ├── Numerical Analysis                          
    └── Theoretical Computer Science                                     
    Veterinary/
    ├── Small Animals    
    ├── Equine           
    └── Food Animals                                      
    Arts_and_Humanities/
    ├── Philosophy                           
    ├── Religious Studies                    
    ├── Literature and Literary Theory       
    ├── History                              
    ├── History and Philosophy of Science    
    ├── Conservation                         
    ├── Archeology (arts and humanities)     
    ├── Visual Arts and Performing Arts       
    ├── Philosophy                            
    ├── Music                                 
    ├── History                               
    ├── Visual Arts and Performing Arts       
    └── Museology                             
    Economics_Econometrics_and_FinanceFinance                       /
    ├── Economics and Econometrics    
    └── Finance   
