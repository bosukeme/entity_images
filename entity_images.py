import requests
import os
import cv2
import numpy as np
import urllib.request
from pymongo import MongoClient
from decouple import config as env_config

VALUE_SERP = env_config("VALUE_SERP")

MONGO_URL_1 = env_config("MONGO_URL_1")
MONGO_URL_2 = env_config("MONGO_URL_2")

client_1 = MongoClient(MONGO_URL_1) ## Ayo's db url
client_2 = MongoClient(MONGO_URL_2) ## My db url

db_1 = client_1.blovids ## Ayo's db 
db_2 = client_2.twitter_users ## My db

collection_0 = db_1.entity_mentions ## Ayo's collection

collection_1 = db_2.twitter_details_collections ## verified handles db
collection_2 = db_2.entity_db



def get_urls_from_search_term_modified(entity, num_results, VALUE_SERP):
    # set up the request parameters
    params_1 = {
        'api_key': VALUE_SERP,
        'q': "site:instagram.com intext:" + entity['entity'],
        'num': '%s' % num_results
    }

    params_2 = {
        'api_key': VALUE_SERP,
        'q': "site:twitter.com intext:" + entity['entity'],
        'num': '%s' % num_results
    }


    if entity['type'] == "Person":
        params_3 = {
            'api_key': VALUE_SERP,
            'search_type': 'images',
            'q':  entity['entity'] +" person",
            'num': '%s' % num_results
        }
    else:
        params_3 = {
            'api_key': VALUE_SERP,
            'search_type': 'images',
            'q':  entity['entity'],
            'num': '%s' % num_results
        }

    
    # make the http GET request to VALUE SERP
    instagram_api_result = requests.get('https://api.valueserp.com/search', params_1)
    twitter_api_result = requests.get('https://api.valueserp.com/search', params_2)
    image_api_result = requests.get('https://api.valueserp.com/search', params_3)
    
    return instagram_api_result.json(), twitter_api_result.json(), image_api_result.json()



def calculate_image_sharpness(image):
    """
    This function takes in an image as a numpy array and then calculates
    a sharpness score (likely based on the density of the pixels?)
    """
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    gnorm = np.sqrt(laplacian**2)
    sharpness = np.average(gnorm)
    
    return sharpness


def convert_image_url_to_array(image_url):
    temp_image_path = "temp_file"
    import requests
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:71.0) Gecko/20100101 Firefox/71.0'}
    r = requests.get(image_url, headers=headers)
    with open(temp_image_path, 'wb') as f:
        f.write(r.content)
    image = cv2.imread(temp_image_path)
    image = cv2.cvtColor(np.uint8(image), cv2.COLOR_BGR2RGB)
    return image


def filter_by_sharpness(image_links):
    try:
        image_arrays = []
        for image_url in image_links:
            image_array = convert_image_url_to_array(image_url)
            _dict = {"image_url" :image_url,
                    "image_array": image_array}
            
            image_arrays.append(_dict)
    except Exception as e:
        print(e)
        pass


    try:
        images = []
        for image_item in image_arrays:
            image_sharpness = calculate_image_sharpness(image_item['image_array'])
            if image_sharpness > 5:
                images.append(image_item)

        images = [a['image_url'] for a in images]
    except:
        images = []
        
    return images



def find_frontal_face(image_links):
    image_links = filter_by_sharpness(image_links)
    
    images_with_one_face = []
    for image_link in image_links:
        try:
            #### read the image url
            # req = urllib.request.urlopen(image_link)
            # arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            # img = cv2.imdecode(arr, -1) # 'Load it as it is'
            img = convert_image_url_to_array(image_link)

            ### convert image to gray scale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ### search for frontal face 
            faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.3,
                    minNeighbors=3,
                    minSize=(30, 30)
            )

            # print("Found {0} Faces!".format(len(faces)))

            if len(faces) == 1:
                try:
                
                    height_width = faces.tolist()[0][2:]
                    resolution = np.prod(height_width)

                except:
                    height_width = list(faces)
                    resolution = np.prod(height_width)

                if resolution / np.prod(list(gray.shape)) > 0.01:
                    images_with_one_face.append({"image":image_link, "resolution":resolution})

        except:
            pass

    images_with_one_face = sorted(images_with_one_face, key=lambda k: k['resolution'], reverse=True)
    
    return images_with_one_face


def find_other_objects(image_links):
    image_links = filter_by_sharpness(image_links)

    one_object = []
    for image_link in image_links:

        try:
            # req = urllib.request.urlopen(img)
            # arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            # image = cv2.imdecode(arr, -1) # 'Load it as it is'
            img = convert_image_url_to_array(image_link)

            # Convert image in grayscale
            gray_im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Contrast adjusting with gamma correction y = 1.2

            gray_correct = np.array(255 * (gray_im / 255) ** 1.2 , dtype='uint8')

            # Contrast adjusting with histogramm equalization
            gray_equ = cv2.equalizeHist(gray_im)

            thresh = cv2.adaptiveThreshold(gray_correct, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 19)
            thresh = cv2.bitwise_not(thresh)
            

            # Dilatation et erosion
            kernel = np.ones((15,15), np.uint8)
            img_dilation = cv2.dilate(thresh, kernel, iterations=1)
            img_erode = cv2.erode(img_dilation,kernel, iterations=1)
            # clean all noise after dilatation and erosion
            img_erode = cv2.medianBlur(img_erode, 7)
            

            # Labeling

            ret, labels = cv2.connectedComponents(img_erode)
            label_hue = np.uint8(179 * labels / np.max(labels))
            blank_ch = 255 * np.ones_like(label_hue)
            labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
            labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
            labeled_img[label_hue == 0] = 0

            resolution = np.prod(gray_im.shape)

            if ret <= 10:
                one_object.append({"image":img, "resolution":resolution})
        except:
            pass
    
    one_object = sorted(one_object, key=lambda k: k['resolution'], reverse=True)
    
    return one_object



def get_record_details(search_dict, collection, find_one=True):
    try:
        query = collection.find_one(search_dict) if find_one else collection.find(search_dict)
        return query
    except Exception as e:
        print(e)
        return None


def insert_records(collection, record):
    try:
        collection.insert_one(record)
    except Exception as e:
        print(e)

def save_to_mongo_db(data):
    insert_records(collection_2, data)
    cur = collection_2.count_documents({})
    print(f"we have {cur} entries")



def get_username_from_search(entity):

    try:
        instagram_data, twitter_data, image_links_result = get_urls_from_search_term_modified(entity, 30, VALUE_SERP)

        try:
            instagram_username = instagram_data['organic_results'][0]['link'].split("/")[3]
        except:
            instagram_username = "NA"

        try:
            twitter_username = twitter_data['organic_results'][0]['link'].split("/")[3]
        except:
            twitter_username = "NA"


        search_dict = {'handle': twitter_username}
        query = get_record_details(search_dict, collection_1, find_one=True)
        if query != None:
            try:
                image_links = [a['image'] for a in image_links_result["image_results"]][:30]
            except:
                image_links = []

            if entity['type'] == "Person":
                images_with_one_face = find_frontal_face(image_links)

                top_images = [a['image'] for a in images_with_one_face]

            else:
                one_object = find_other_objects(image_links)
                top_images = [a['image'] for a in one_object]

            data = {
                "Entity": entity['entity'],
                "instagram_handle": instagram_username,
                "twitter_handle": twitter_username,
                "top_images": top_images
            }

            print(data)


            search_dict = {'handle': twitter_username}
            query = get_record_details(search_dict, collection_1, find_one=True)

            if query:
                save_to_mongo_db(data)


        else:
            pass
        #os.system(f"instalooter user {username}  --videos-only --num-to-dl 5 outputs/{entity_dir}")
    except Exception as e:
        print(e)


def search_db():
    
    entities_list = list(collection_0.find({},{ "_id": 0, "entity": 1, "type":1})) 
    #entities_list = list((val for dic in entities_list for val in dic.values()))

    return entities_list


def run_script():
    entities_list = search_db()
    
    for entity in entities_list:
        print(entity)
        search_dict = {'Entity': entity['entity']}
        query = get_record_details(search_dict, collection_2, find_one=True)

        if query == None:
            get_username_from_search(entity)

run_script()
