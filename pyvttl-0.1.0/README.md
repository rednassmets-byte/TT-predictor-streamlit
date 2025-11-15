
# pyvttl
Python Wrapper for VTTL API, as documented [in the TabT pages](https://api.frenoy.net/group__TabTAPIfunctions.html). Please refer to these pages to interpret the results from each method described below.

## VttlApi Methods

### Initialization
```python
from pyvttl.vttl_api import VttlApi
api = VttlApi(username="your_username", password="your_password")
```

### Public Methods
- `getSeasons()` - Get list of seasons
- `getClubs(season=None, province=None, club=None, **kwargs)` - Get club list
- `getClubTeams(club, season=None, **kwargs)` - Get teams for a club
- `getDivisionRanking(divisionId=None, week=None, season=None, **kwargs)` - Get division ranking
- `getDivisions(level=None, season=None, show_division_name=None, **kwargs)` - Get division list
- `getMatches(division_id=None, club=None, team=None, division_category=None, season=None, week_name=None, level=None, show_division_name=None, year_date_from=None, year_date_to=None, with_details=None, match_id=None, match_unique_id=None, **kwargs)` - Get matches and results
- `getMatchSystems(**kwargs)` - Get match systems
- `getMembers(club=None, season=None, player_category=None, unique_index=None, name_search=None, extended_information=None, ranking_points_information=None, with_results=None, with_opponent_ranking_evaluation=None, **kwargs)` - Get members
- `getTournaments(season=None, tournament_unique_index=None, with_results=None, with_registrations=None, **kwargs)` - Get tournaments
- `getPlayerCategories(season=None, unique_index=None, short_name_search=None, ranking_category=None, **kwargs)` - Get player categories
- `upload(*args, **kwargs)` - Not implemented (read-only)
- `tournamentRegister(*args, **kwargs)` - Not implemented (read-only)

## Province Enum & Methods

```python
from pyvttl.province import Province
```

### Enum Members
- `Province.VLAAMS_BRABANT`
- `Province.BRABANT_WALLON`
- `Province.ANTWERPEN`
- `Province.OOST_VLAANDEREN`
- `Province.WEST_VLAANDEREN`
- `Province.LIMBURG`
- `Province.HAINAUT`
- `Province.LUXEMBOURG`
- `Province.LIEGE`
- `Province.NAMUR`
- `Province.VTTL`
- `Province.AFTT`

### Methods
- `Province.name(enum_name)` - Get display name for enum
- `Province.getVTTLProvinceDefinition(api_instance)` - Get unique Category/CategoryName combinations (class method)

### Example Usage
```python
from pyvttl.vttl_api import VttlApi
from pyvttl.province import Province

api = VttlApi(username="your_username", password="your_password")

# Get matches for a specific club
matches = api.getMatches(club='Vl-B234')
