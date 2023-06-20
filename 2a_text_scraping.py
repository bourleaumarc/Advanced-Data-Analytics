import csv
import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from tqdm import tqdm
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from urllib3.util.url import get_host
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Create an empty pandas DataFrame to store the watch data: we will then add all the values that are stored in the watch_data_dict
watch_data = pd.DataFrame(
    columns=["Brand", "Family", "Reference", "Name", "Movement", "Produced", "Limited", "Materials", "Material",
             "Bezel", "Glass", "Back", "Shape", "Diameter", "Height", "Lug Width", "W/R", "Nickname", "Color", "Finish",
             "Indexes", "Hands", "Price"])

# Brands and their corresponding URLs
brands = ["Rolex", "Omega", "Patek-philippe", "Audemars-piguet", "Cartier", "Breitling", "Tag-Heuer", "Iwc", "Chopard",
         "Jaeger-lecoultre", "Blancpain", "Hublot", "Zenith", "Certina", "Panerai","Girard-perregaux", "Breguet",
          "Montblanc", "Tissot", "Bulgari"]
urls = ["https://watchbase.com/" + brand.lower() for brand in brands]

# Cleaning the data: sometimes the variable Material is saved in plural. We added both in the dataframe watch data and we merged them into "Materials".
watch_data['merged_column'] = watch_data['Materials'].fillna(watch_data['Material'])
watch_data.drop(['Materials', 'Material'], axis=1, inplace=True)
watch_data.rename(columns={'merged_column': 'Materials'}, inplace=True)

# Open the browser
driver = webdriver.Firefox()

# Loop through each brand's page
for url in urls:
    driver.get(url)

    # Find the first 5 categories on the page and get their links
    categories = driver.find_elements(By.CSS_SELECTOR, ".family-box.row .col-md-6 h2.title a")[:5]
    category_links = [category.get_attribute("href") for category in categories]

    # Loop through each category and extract the text
    for link in tqdm(category_links):
        driver.get(link)

        # Wait for page to load
        time.sleep(2)

        # Scroll down the page several times to trigger image loading
        for i in range(5):
            driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
            time.sleep(2)

        # Now we landed on the category website but we have to be inside each watch website
        watches = driver.find_elements(By.CSS_SELECTOR, ".watch-block-container a.item-block")[:50]  # we seek to take 50 images
        watches_link = [watch.get_attribute("href") for watch in watches]

        for each_watch in watches_link:
            driver.get(each_watch)

            # Wait for page to load
            time.sleep(2)

            # Scroll down the page several times to trigger image loading
            for i in range(5):
                driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
                time.sleep(2)

            watch_data_dict = {}

            # Extract watch details from the first table
            first_table = driver.find_element(By.CLASS_NAME, "info-table")
            rows_first_table = first_table.find_elements(By.TAG_NAME, "tr")

            for row in rows_first_table:
                row_data = row.text.split(":", 1)  # split on first occurrence of ":"

                if len(row_data) == 2:
                    key = row_data[0].strip()
                    value = row_data[1].strip()
                    watch_data_dict[key] = value
                    print(watch_data_dict)

            # Extract watch details from the second table
            second_table = driver.find_element(By.CLASS_NAME, "col-xs-6")
            row_second_table = second_table.find_elements(By.TAG_NAME, "tr")

            for row in row_second_table:
                row_data = row.text.split(":", 1)  # because reference sometimes has (aka number)

                if len(row_data) == 2:
                    key = row_data[0].strip()
                    value = row_data[1].strip()
                    watch_data_dict[key] = value
                    print(watch_data_dict)

            try:
                # Find the element with the "data-url" attribute
                element = driver.find_element(By.CSS_SELECTOR, "#pricechart")
                data_url = element.get_attribute("data-url")

                # Open the "data-url" in a new tab
                driver.execute_script(f"window.open('{data_url}')")

                # Switch to the new tab
                # Add an explicit wait to ensure the new tab is fully loaded before switching
                WebDriverWait(driver, 10).until(EC.number_of_windows_to_be(2))
                driver.switch_to.window(driver.window_handles[1])

                # Wait for the new tab to load
                # Add an explicit wait to ensure the content of the new tab is loaded
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR,"#\\/datasets\\/0\\/data\\/1 > td:nth-child(2) > span:nth-child(1) > span:nth-child(1)")))

                # Find the price
                price_element = driver.find_element(By.CSS_SELECTOR, "#\\/datasets\\/0\\/data\\/1 > td:nth-child(2) > span:nth-child(1) > span:nth-child(1)")
                price_value = price_element.text if price_element.text != "null" else "NA"

                # Add the price value to the watch_data_dict
                watch_data_dict["Price"] = price_value

                # Close the new tab
                driver.close()

                # Switch back to the main tab
                driver.switch_to.window(driver.window_handles[0])

            except NoSuchElementException:
                    print("No price section found. Skipping watch.")
            # Add the watch data to the pandas DataFrame
            watch_data.loc[len(watch_data)] = watch_data_dict
            print(watch_data)

# Replace "NaN" values in the "Price" column with "NA"
watch_data["Price"].fillna("NA", inplace=True)
# Count the number of missing values
for col in watch_data.columns:
    missing_values = watch_data[col].isna().sum()
    print(f"Number of missing {col} values: {missing_values}")

# Save the pandas DataFrame to a CSV file
watch_data.to_csv("2b_cleaned_watch_text_def.csv", index=False)
