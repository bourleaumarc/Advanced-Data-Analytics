from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import os
import urllib.request
import selenium.common.exceptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Brands and their corresponding URLs
brands = ["Rolex", "Omega", "Patek-philippe", "Audemars-piguet", "Cartier", "Breitling", "Tag-Heuer", "Iwc", "Chopard",
          "Jaeger-lecoultre", "Blancpain", "Hublot", "Zenith", "Certina", "Panerai", "Girard-perregaux", "Breguet",
          "Montblanc", "Tissot", "Bulgari"]

urls = ["https://watchbase.com/" + brand.lower() for brand in brands]

# Number of images to download per category per brand
images_per_category = 55
target_per_category = 50
timeToSleep = 2

# Open the browser
driver = webdriver.Firefox()

# Loop through each brand's page
for url in urls:
    driver.get(url)

    # Find the first 5 categories on the page and get their links
    categories = driver.find_elements(By.CSS_SELECTOR, ".family-box.row .col-md-6 h2.title a")[:5]
    category_links = [category.get_attribute("href") for category in categories]

    # Loop through each category and extract the image URLs
    total_count = 0
    for link in category_links:
        print("-------------------" + link)
        try:
            driver.get(link)
        except selenium.common.exceptions.NoSuchWindowException:
            # Reopen the window or tab
            driver.execute_script("window.open('');")
            driver.switch_to.window(driver.window_handles[-1])
            # Navigate to the URL again
            driver.get(link)

        # Wait for page to load
        time.sleep(timeToSleep)

        # Scroll down the page several times to trigger image loading
        for i in range(5):
            driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
            time.sleep(timeToSleep)

        # Wait for all images to load
        WebDriverWait(driver, 10).until(EC.visibility_of_all_elements_located((By.TAG_NAME, 'img')))
        # Find all the image tags on the page
        images = driver.find_elements(By.TAG_NAME, 'img')[:images_per_category]
        # Find the name of the references
        reference_links = driver.find_elements(By.TAG_NAME, "strong")
        reference_names = [reference.text for reference in reference_links]
        # Collect the URLs of the images that match the expected prefix
        vector = ["speedmaster-automatic-and-other", "datejust"]
        brand_prefix = f"https://cdn.watchbase.com/watch/md/{link.split('/')[-2] + '/' + link.split('/')[-1]}"
        image_urls = [image.get_attribute('src') for image in images if
                      image.get_attribute('src').startswith(brand_prefix) or
                      (image.get_attribute('src').startswith(v) for v in vector)]
        # Replace "md" with "lg" in each URL to get higher quality images
        modified_urls = [url.replace('/md/', '/lg/') for url in image_urls]
        # Create a folder on your desktop to save the images
        folder_path = os.path.expanduser(f"~/Desktop/{link.split('/')[-2]}_images")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Download the images and save them to the folder with the reference name as file name
        count = 0
        for i, url in enumerate(modified_urls):
            if count == target_per_category:
                break

            reference_name = reference_names[(i - 3) % len(reference_names)]
            if '/' in reference_name:
                reference_name = reference_name.replace('/', '-')
            file_name = os.path.join(folder_path, f"{reference_name}.jpg")
            urllib.request.urlretrieve(url, file_name)
            print(f"Downloaded image {i + 1}: {file_name}")
            count += 1
            total_count += 1

        if total_count >= images_per_category * len(category_links):
            break

# Close the browser
driver.quit()
