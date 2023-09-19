import requests
from io import StringIO
import pandas as pd
from mosek.fusion import *
from bs4 import BeautifulSoup
import re
pd.set_option('display.max_columns', None)

def Read_DIMACS_Graph(instance_name):
    # Fetch DIMACS data from the web
   
    # Trick
    #url = f"https://mat.tepper.cmu.edu/COLOR/instances/{instance_name}"
    # M. Caramia
    url= f"https://raw.githubusercontent.com/dynaroars/npbench/master/instances/coloring/graph_color/{instance_name}"
    response = requests.get(url)
    data = StringIO(response.text)
    # Initialize an empty adjacency list
    adjacency_list = {}

    for line in data:
        # Strip leading/trailing whitespace
        line = line.strip()

        # DIMACS lines start with a character that denotes the type of line.
        if line.startswith('c'):
            continue
        elif line.startswith('p'):
            _, _, vertices, _ = line.split()
            for i in range(1, int(vertices) + 1):
                adjacency_list[i] = []
        elif line.startswith('e'):
            _, start, end = line.split()
            adjacency_list[int(start)].append(int(end))
            adjacency_list[int(end)].append(int(start))
    return adjacency_list

def Get_Instance_Metadata(instance_name,Name):
    if Name == 'Trick':
        # Fetch the data from the webpage
        url = "https://mat.tepper.cmu.edu/COLOR/instances.html"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # This is a pattern to find the filename, nodes, edges, optimal coloring, and source.
        pattern = r'<a href="instances/([^"]+)">\s+([^<]+)\s+</a>\s+\((\d+),(\d+)\),\s+([^<]+),\s+<a href="#[^"]+">([^<]+)</a>'
        matches = re.findall(pattern, str(soup))

        # Extract values from the regex matches and filter for the desired instance name
        data = [{'file': match[0],
                'name': match[1].strip(),
                'nodes': int(match[2]),
                'edges': int(match[3]),
                'optimal_coloring': int(match[4].strip()),
                'source': match[5]} for match in matches if match[1].strip() == instance_name]

        return data[0] if data else None
    else:     
        url = "https://dynaroars.github.io/npbench/graphcoloring.html"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Fetching the relevant table (assuming it's the first table on the page)
        table = soup.find_all('table')[0]
        
        # Create an empty list to collect all the rows
        data = []
        
        # Iterate through each row of the table
        for row in table.find_all('tr'):
            columns = row.find_all('td')
            
            # Ensure the row has enough columns
            if len(columns) >= 6:
                file = columns[1].text.strip()
                name = columns[1].text.strip()  # Assuming the name is also in the second column
                nodes = columns[2].text.strip()
                edges = columns[3].text.strip()
                optimal_coloring = columns[4].text.strip()
                source = columns[5].text.strip()
                
                # Append the data to the list
                data.append({
                    'file': file,
                    'name': name,
                    'nodes': int(nodes) if nodes != '?' else None,
                    'edges': int(edges) if edges != '?' else None,
                    'optimal_coloring': int(optimal_coloring) if optimal_coloring != '?' else None,
                    'source': source
                })

        # Filter the list for rows that match the instance_name
        data = [row for row in data if row['name'] == instance_name]
        
        # Return the first match if found, else None
        return data[0] if data else None