from enum import Enum

class Province(Enum):
    VLAAMS_BRABANT = 2
    BRABANT_WALLON = 3
    ANTWERPEN = 4
    OOST_VLAANDEREN = 5
    WEST_VLAANDEREN = 6
    LIMBURG = 7
    HAINAUT = 8
    LUXEMBOURG = 9
    LIEGE = 10
    NAMUR = 11
    VTTL = 12
    AFTT = 13

    _name_map = {
        'VLAAMS_BRABANT': "Vlaams-Brabant & Br.",
        'BRABANT_WALLON': "Br. & Brabant Wallon",
        'ANTWERPEN': "Antwerpen",
        'OOST_VLAANDEREN': "Oost-Vlaanderen",
        'WEST_VLAANDEREN': "West-Vlaanderen",
        'LIMBURG': "Limburg",
        'HAINAUT': "Hainaut",
        'LUXEMBOURG': "Luxembourg",
        'LIEGE': "Liège",
        'NAMUR': "Namur",
        'VTTL': "VTTL",
        'AFTT': "AFTT"
    }

    @classmethod
    def name(cls, enum_name):
        return cls._name_map.get(enum_name, None)

    @classmethod
    def getVTTLProvinceDefinition(self, api_instance):
        """
        Print unique Category/CategoryName combinations from clubs. This is basically
        how we generated the above enum...
        Args:
            api_instance (VttlApi): An instance of VttlApi to call getClubs().
        """
        clubs_result = api_instance.getClubs()
        unique_categories = set()
        for club in clubs_result.ClubEntries:
            category = getattr(club, 'Category', None)
            category_name = getattr(club, 'CategoryName', None)
            if category is not None and category_name is not None:
                unique_categories.add((category, category_name))
        return unique_categories
