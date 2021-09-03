import requests
import os
import instalooter
import cv2
import numpy as np
import urllib.request
from pymongo import MongoClient
from decouple import config as env_config

SERP_VALUE = env_config("SERP_VALUE")

MONGO_URL_1 = env_config("MONGO_URL_1")
MONGO_URL_2 = env_config("MONGO_URL_2")

client_1 = MongoClient(MONGO_URL_1) ## Ayo's db
client_2 = MongoClient(MONGO_URL_2) ## My db

db_1 = client_1.blovids
db_2 = client_2.twitter_users

collection_0 = db_1.entity_mentions

collection_1 = db_2.twitter_details_collections ## verified handles db
collection_2 = db_2.entity_db



def get_urls_from_search_term_modified(entity, num_results, VALUE_SERP):
    # set up the request parameters
    params_1 = {
        'api_key': VALUE_SERP,
        'q': "site:instagram.com intext:" + entity,
        'num': '%s' % num_results
    }

    params_2 = {
        'api_key': VALUE_SERP,
        'q': "site:twitter.com intext:" + entity,
        'num': '%s' % num_results
    }

    params_3 = {
        'api_key': VALUE_SERP,
        'search_type': 'images',
        'q':  entity +" person",
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
    """
    This takes an image URL and converts it to a numpy array of the image
    """
    req = urllib.request.urlopen(image_url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    image = cv2.imdecode(arr, -1) # 'Load it as it is'

    return image


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
        entity_dir = "_".join(entity.split(" "))
        instagram_data, twitter_data, image_links_result = get_urls_from_search_term_modified(entity, 30, VALUE_SERP)

        try:
            instagram_username = instagram_data['organic_results'][0]['link'].split("/")[3]
            print(instagram_username)
        except:
            instagram_username = "NA"

        try:
            twitter_username = twitter_data['organic_results'][0]['link'].split("/")[3]
            print(twitter_username)
        except:
            twitter_username = "NA"

        try:
            image_links = [a['image'] for a in image_links_result["image_results"]][:20]
            print(image_links[:1])
        except:
            image_links = []

        try:
            #image_arrays = [convert_image_url_to_array(image_url) for image_url in image_links]

            image_arrays = []
            for image_url in image_links:
                image_array = convert_image_url_to_array(image_url)
                _dict = {"image_url" :image_url,
                        "image_array": image_array}
                
                image_arrays.append(_dict)
        except:
            image_arrays = []

        try:
            images = []
            for image_item in image_arrays:
                image_sharpness = calculate_image_sharpness(image_item['image_array'])
                if image_sharpness > 1:
                    images.append(image_item)
            
            images = [a['image_url'] for a in images]
        except:
            images = []


        data = {
            "Entity": entity,
            "instagram_handle": instagram_username,
            "twitter_handle": twitter_username,
            "top_images": images
        }

        print(data)


        search_dict = {'handle': twitter_username}
        query = get_record_details(search_dict, collection_1, find_one=True)

        if query:
            save_to_mongo_db(data)

        #os.system(f"instalooter user {username}  --videos-only --num-to-dl 5 outputs/{entity_dir}")
    except Exception as e:
        print(e)


def search_db():
    
    entities_list = list(collection_0.find({},{ "_id": 0, "entity": 1})) 
    entities_list = list((val for dic in entities_list for val in dic.values()))

    return entities_list


def run_script():
    entities_list = search_db()

    for entity in entities_list:
        search_dict = {'Entity': entity}
        query = get_record_details(search_dict, collection_2, find_one=True)

        if query == None:
            get_username_from_search(entity)

run_script()
