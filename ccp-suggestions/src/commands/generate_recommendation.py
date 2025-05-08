import cv2
import json
import numpy as np
import pandas as pd
from google.cloud import storage
from itertools import combinations

from src.static.constants import bucket_name, bucket_image_folder


class GenerateRecommendations:
    def __init__(self):
        self.recommendations = []

    def download_filtered_images(self):
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        data = []
        for i in range(5):
            img_name = f"frame_{i}.jpg"
            meta_name = f"frame_metadata_{i}.json"

            img_blob = bucket.blob(f"{bucket_image_folder}/{img_name}")
            meta_blob = bucket.blob(f"{bucket_image_folder}/{meta_name}")

            image_bytes = img_blob.download_as_bytes()
            np_arr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            json_bytes = meta_blob.download_as_bytes()
            metadata = json.loads(json_bytes.decode("utf-8"))

            data.append({"image": image, "metadata": metadata})

        return data

    def analyze_size_parity(self, metadata):
        """
        This method gives a score for a set of criteria based on product density.
        A good density of products is based on distribution of the boxes and how many of them overlap between each other.
        The model is trained to recognize products on the front, so it shouldn't have much cluttering.
        """
        df_metadata = pd.DataFrame(metadata)
        df_metadata["length"] = df_metadata["x2"] - df_metadata["x1"]
        df_metadata["height"] = df_metadata["y2"] - df_metadata["y1"]

        mean_height = df_metadata["height"].mean()
        var_height = df_metadata["height"].var()
        height_score = (mean_height - var_height) / (mean_height + var_height)

        if height_score >= 0.9:
            self.recommendations.append(
                "La altura de los productos es muy consistente. Esto significa que los clientes tendrán una percepción más positiva de los espacios en su tienda al garantizar una homogeneidad dimensional"
            )
        elif height_score < 0.9 and height_score >= 0.5:
            self.recommendations.append(
                "La altura de los productos tiene una variabilidad moderada. Para este caso, se recomienda agrupar productos con alturas y categorías similares. Si es posible, traslade los productos de menor rotación a otro espacio y consiga un producto de un tamaño y categoría similar"
            )
        elif height_score < 0.5:
            self.recommendations.append(
                "La altura de los productos no es suficientemente consistente. Esto significa que los productos que está mostrando probablemente tienen propósitos distintos. Es muy importante mantener espacios homogéneos con productos similares para mejorar la experiencia de los compradores"
            )

        mean_length = df_metadata["length"].mean()
        var_length = df_metadata["length"].var()
        length_score = (mean_length - var_length) / (mean_length + var_length)

        if length_score >= 0.9:
            self.recommendations.append(
                "El ancho de los productos es muy consistente. Los clientes aprecian las tiendas que utilizan los anchos de las estanterías forma correcta ya que se ven más completas"
            )
        elif length_score < 0.9 and height_score >= 0.5:
            self.recommendations.append(
                "El ancho de los productos es variable. Esto significa que tendrá que ser creativo en la organización de los productos y utilizar los productos más delgados para las esquinas de las estanterías. No olvide garantizar que los tipos de productos se mantengan juntos"
            )
        elif length_score < 0.5:
            self.recommendations.append(
                "El ancho de los productos no es consistente. Debe intentar trasladar productos entre estanterías para mejorar la distribución de los productos en su tienda. Es muy importante que los clientes no tengan la percepción de que los espacios están vacíos"
            )

    def analyze_spread(self, metadata):
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

        if spread_score >= 0.9:
            self.recommendations.append(
                "La distribución de los productos es muy buena. Una estantería agradable debería tener pocos espacios disponibles en la parte frontal"
            )
        elif spread_score < 0.9 and spread_score >= 0.5:
            self.recommendations.append(
                "La distribución de los productos es mejorable. Se detectó que los productos en la imagen no están lo suficientemente cerca entre ellos, o no están organizados en su espacio. Si tiene espacios vacíos, es necesario adquirir productos que le permitan suplir la necesidad"
            )
        elif spread_score < 0.5:
            self.recommendations.append(
                "La distribución de los productos es mala. Se encuentran muy pocos productos en un espacio muy grande. Esto significa que se está desperdiciando el espacio disponible para almacenar, o que necesita adquirir más productos para la demanda que está experimentando"
            )
    
    def calculate_overlapping_areas(self, metadata):
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

        if overlapping_score > 0.3:
            self.recommendations.append(
                "Se detectó que muchos productos se solapan entre ellos. Esto significa que su espacio no está organizado de forma correcta. Valide que cuenta con los productos necesarios para que los productos traseros de su espacio no sean inmediatamente visibles"
            )
        elif overlapping_score <= 0.3 and overlapping_score > 0.1:
            self.recommendations.append(
                "La organización de su espacio puede mejorar. El espacio cuenta con un orden general, sin embargo, los productos traseros son visibles. Esto puede significar que debe reorganizar teniendo en cuenta el tamaño, o agregar nuevos productos que cumplan la función de cubrir lo espacios vacíos"
            )
        elif overlapping_score <= 0.1:
            self.recommendations.append(
                "La distribución de los productos es buena. No se observa un solapamiento entre productos apreciable. Esto significa que sus espacios son agradables y que la busqueda de productos en su tienda es ágil y fácil de reemplazar"
            )

    def execute(self):
        data = self.download_filtered_images()
        for image in data:
            self.analyze_size_parity(image["metadata"])
            self.analyze_spread(image["metadata"])
            self.calculate_overlapping_areas(image["metadata"])

        return self.recommendations