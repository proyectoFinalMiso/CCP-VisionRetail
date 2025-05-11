import cv2
import json
import numpy as np
import pandas as pd
from google.cloud import storage
from itertools import combinations

from src.static.constants import bucket_name, bucket_image_folder, email_topic
from src.commands.common.pubsub import publish_message


class GenerateRecommendations:
    def __init__(self, body):
        self.body = body
        self.recommendations = []

    def download_filtered_images(self):
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        data = []
        for img_set in self.body['message']:
            img_name = img_set['image']
            meta_name = img_set['metadata']

            img_blob = bucket.blob(f"{bucket_image_folder}/{img_name}")
            meta_blob = bucket.blob(f"{bucket_image_folder}/{meta_name}")

            image_bytes = img_blob.download_as_bytes()
            np_arr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            json_bytes = meta_blob.download_as_bytes()
            metadata = json.loads(json_bytes.decode("utf-8"))

            data.append({"name": img_name, "image": image, "metadata": metadata})

        return data

    def analyze_size_parity(self, metadata, name: int):
        """
        This method gives a score for a set of criteria based on product density.
        A good density of products is based on distribution of the boxes and how many of them overlap between each other.
        The model is trained to recognize products on the front, so it shouldn't have much cluttering.
        """
        df_metadata = pd.DataFrame(metadata)
        df_metadata["length"] = df_metadata["x2"] - df_metadata["x1"]
        df_metadata["height"] = df_metadata["y2"] - df_metadata["y1"]

        mean_height = df_metadata["height"].mean()
        var_height = df_metadata["height"].std()
        height_score = (mean_height - var_height) / (mean_height + var_height)

        if height_score >= 0.9:
            self.recommendations.append(
                {"name": name, "message": "La altura de los productos es muy consistente y están bien agrupados. Esto significa que los clientes tendrán una percepción más positiva de los espacios en su tienda ya que podrán encontrar los productos más facilmente"}
            )
        elif height_score < 0.9 and height_score >= 0.5:
            self.recommendations.append(
                {"name": name, "message": "Su espacio tiene productos con alturas similares, pero podría mejorar. Se recomienda agrupar productos con alturas y categorías similares. Si es posible, traslade los productos de menor rotación a otro espacio y consiga un producto de un tamaño y categoría similar"}
            )
        elif height_score < 0.5:
            self.recommendations.append(
                {"name": name, "message": "La altura de los productos en su espacio es excesivamente diferente. Esto significa que los productos que está mostrando probablemente tienen propósitos distintos. Es muy importante mantener espacios homogéneos con productos similares para que sus clientes no tengan dificultades en encontrar lo que buscan"}
            )

        mean_length = df_metadata["length"].mean()
        var_length = df_metadata["length"].std()
        length_score = (mean_length - var_length) / (mean_length + var_length)

        if length_score >= 0.9:
            self.recommendations.append(
                {"name": name, "message": "Los productos están bien agrupados por ancho y usan el espacio adecuadamente. Los clientes aprecian las tiendas organizadas en horizontal ya que las hace ver más abundantes"}
            )
        elif length_score < 0.9 and height_score >= 0.5:
            self.recommendations.append(
                {"name": name, "message": "Debe trabajar en la organización de los productos para garantizar que los productos estén bien distribuidos por ancho. Utilice los productos más delgados para las esquinas de las estanterías. Recuerde que los propósitos de los productos sean los mismos"}
            )
        elif length_score < 0.5:
            self.recommendations.append(
                {"name": name, "message": "El ancho de los productos es excesivamente diferente. Debe organizar sus espacios para que productos de tamaño y propósito similar se mantengan en posiciones cercanas. Es muy importante que los clientes no tengan la percepción de que los espacios están vacíos"}
            )

    def analyze_spread(self, metadata, name: int):
        """
        This method gives a score for a set of criteria based on product spread.
        A good density of products is based on distribution of the boxes and how much the total surface area of the products fill the
        actual area of the entire distribution of products. A good spread is expected to occupy near 95% of the total area. It also
        takes into account overlapping between products areas and what % the overlap represents
        """
        df_metadata = pd.DataFrame(metadata)
        df_metadata["length"] = df_metadata["x2"] - df_metadata["x1"]
        df_metadata["height"] = df_metadata["y2"] - df_metadata["y1"]
        df_metadata["area"] = df_metadata["height"] * df_metadata["length"]

        max_x1 = df_metadata["x1"].min()
        max_x2 = df_metadata["x2"].max()
        max_y1 = df_metadata["y1"].min()
        max_y2 = df_metadata["y2"].max()
        total_spread_area = (max_x2 - max_x1) * (max_y2 - max_y1)
        total_surface_area = df_metadata["area"].sum()

        spread_score = total_surface_area / total_spread_area
        print(spread_score)

        if spread_score < 1 and spread_score >= 0.8:
            self.recommendations.append(
                {"name": name, "message": "Sus espacios están eficientemente distribuidos ya que utliza la mayoría del espacio disponible para ubicar sus productos"}
            )
        elif spread_score < 0.8 and spread_score >= 0.5:
            self.recommendations.append(
                {"name": name, "message": "Se puede mejorar el uso del espacio. Se detectó que los productos en la imagen no están lo suficientemente cerca entre ellos, o no están organizados en su espacio. Si tiene espacios vacíos, es necesario adquirir productos que le permitan suplir la necesidad"}
            )
        elif spread_score < 0.5:
            self.recommendations.append(
                {"name": name, "message": "El uso del espacio no es el adecuado. Se encuentran muy pocos productos en un espacio muy grande. Esto significa que se está desperdiciando el espacio disponible para almacenar, o que necesita adquirir más productos para la demanda que está experimentando"}
            )
        elif spread_score >= 1:
            self.recommendations.append(
                {"name": name, "message": "Tiene un exceso de productos en su espacio. Esto significa que sus productos están muy acumulados, provocando que el espacio tenga un efecto negativo sobre los clientes"}
            )
    
    def calculate_overlapping_areas(self, metadata, name: int):
        """
        This method gives a score based on the percentage of overlap present in the image.
        The algorithm shouldn't detect a significant overlap as that means that the products are cluttered.
        A good distribution should be close to 0% of overlap
        """
        df_metadata = pd.DataFrame(metadata)
        df_metadata["length"] = df_metadata["x2"] - df_metadata["x1"]
        df_metadata["height"] = df_metadata["y2"] - df_metadata["y1"]
        df_metadata["area"] = df_metadata["height"] * df_metadata["length"]

        max_x1 = df_metadata["x1"].min()
        max_x2 = df_metadata["x2"].max()
        max_y1 = df_metadata["y1"].min()
        max_y2 = df_metadata["y2"].max()
        total_spread_area = (max_x2 - max_x1) * (max_y2 - max_y1)

        def compute_overlapping_area(rect1, rect2):
            x_overlap = max(0, min(rect1['x2'], rect2['x2']) - max(rect1['x1'], rect2['x1']))
            y_overlap = max(0, min(rect1['y2'], rect2['y2']) - max(rect1['y1'], rect2['y1']))
            return x_overlap * y_overlap
        
        overlapped_areas = []
        for i, j in combinations(df_metadata.index, 2):
            area = compute_overlapping_area(df_metadata.loc[i], df_metadata.loc[j])
            overlapped_areas.append(area)
        
        total_overlapping_area = sum(overlapped_areas)
        overlapping_score = total_overlapping_area / total_spread_area
        print(overlapping_score)

        if overlapping_score > 0.3:
            self.recommendations.append(
                {"name": name, "message": "Se detectó que muchos productos se solapan entre ellos. Esto significa que su espacio no está organizado de forma correcta. Valide que cuenta con los productos necesarios para que los productos traseros de su espacio no sean inmediatamente visibles"}
            )
        elif overlapping_score <= 0.3 and overlapping_score > 0.1:
            self.recommendations.append(
                {"name": name, "message": "La organización de su espacio puede mejorar. El espacio cuenta con un orden general, sin embargo, los productos traseros son visibles. Esto puede significar que debe reorganizar teniendo en cuenta el tamaño, o agregar nuevos productos que cumplan la función de cubrir lo espacios vacíos"}
            )
        elif overlapping_score <= 0.1:
            self.recommendations.append(
                {"name": name, "message": "Los productos se encuentran ordenados correctamente. Esto significa que sus espacios son agradables y que la busqueda de productos en su tienda es ágil y fácil de reemplazar"}
            )

    def execute(self):
        data = self.download_filtered_images()
        for image in data:
            self.analyze_size_parity(image["metadata"], image['name'])
            self.analyze_spread(image["metadata"], image['name'])
            self.calculate_overlapping_areas(image["metadata"], image['name'])
        
        message = {
            "customer": self.body['customer'],
            "seller": self.body['seller'],
            "message": self.recommendations
        }
        
        r = publish_message(email_topic, message)
        return {"response": r, "status_code": 200}