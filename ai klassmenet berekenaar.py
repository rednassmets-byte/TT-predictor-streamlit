#setup
from pyvttl.vttl_api import VttlApi
from pyvttl.province import Province
api = VttlApi(username="your_username", password="your_password")
naam = 'Sander smets'


def get_Results(club: str, name_search: str = , season: int = ,):
    """Call the API and return members for a club."""
    return api.getMembers(
        club=club,
        season=season,
        name_search=name_search,
        extended_information=None,
        ranking_points_information=None,
        with_results=True,
        with_opponent_ranking_evaluation=None,
    )
result = get_Results(club='A-182', name_search='Wannes', season=26)      

print(result)


kaart = {'A':[0,0],'B0':[0,0],'B2':[0,0],'B4':[0,0],'B6':[0,0],'C0':[0,0],'C2':[0,0],'C4':[0,0],'C6':[0,0],'D0':[0,0],'D2':[0,0],'D4':[0,0],'D6':[0,0],'E0':[0,0],'E2':[0,0],'E4':[0,0],'E6':[0,0],'NG':[0,0]}
