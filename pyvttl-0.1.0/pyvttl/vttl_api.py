"""
VttlApi: Python wrapper for the VTTL SOAP API (read-only)

Documentation: https://api.frenoy.net/index.html
"""
import zeep
from pyvttl.province import Province
from pyvttl.vttl_types import ShowDivisionNameType

class VttlApi:
    """
    Python wrapper for the VTTL SOAP API (read-only).

    Example usage:
        api = VttlApi(wsdl_url, club_id, password)
        club_info = api.get_club_info()
    """
    def __init__(self, username: str = None, password: str = None, wsdl_url: str = "https://api.vttl.be/?wsdl"):
        """
        Initialize the VttlApi client.

        Args:
            username (str, optional): Username for authentication (club or user).
            password (str, optional): Password for authentication.
            wsdl_url (str, optional): WSDL URL for the VTTL API. Defaults to "https://api.vttl.be/?wsdl".
        """
        self.wsdl_url = wsdl_url
        self.username = username
        self.password = password
        self.client = zeep.Client(wsdl=self.wsdl_url)
        # Store raw creds; build Zeep typed object only when needed
        self.credentials = {
            'Account': self.username or None,
            'Password': self.password or None,
        }

    def getSeasons(self):
        """
        Get list of seasons.
        Returns:
            dict: Seasons data.
        """
        return self.client.service.GetSeasons()

    def getClubs(self, season: int = None, province: 'Province' = None, club: str = None, **kwargs) -> dict:
        """
        Retrieve club list.
        Args:
            season (int, optional): Season ID.
            province (int, optional): Province ID.
            club (str, optional): Club string.
            **kwargs: Additional request parameters.
        Returns:
            dict: Clubs data.
        """
        request = {}
        if self.username and self.password:
           request['Credentials'] = self.credentials
        if season is not None:
            request['Season'] = season
        if province is not None:
            from pyvttl.province import Province
            if not isinstance(province, Province):
                raise TypeError("province must be of type Province enum")
            request['ClubCategory'] = province.value
        if club is not None:
            request['Club'] = club
        request.update(kwargs)
        if request:
            return self.client.service.GetClubs(**request)
        else:
            return self.client.service.GetClubs()

    def getClubTeams(self, club, season=None, **kwargs):
        """
        Get all teams of a given club.
        Args:
            club (str, mandatory): Club ID filter.
            season (str, optional): Season filter.
            **kwargs: Additional request parameters.
        Returns:
            dict: Club teams data.
        """
        request = {}
        if club:
            request['Club'] = club
        if season:
            request['Season'] = season
        request.update(kwargs)
        if request:
            return self.client.service.GetClubTeams(**request)
        else:
            return self.client.service.GetClubTeams()

    def getDivisionRanking(self, divisionId=None, week=None, season=None, **kwargs):
        """
        Get ranking of a division for a given week.
        Args:
            divisionId (str, optional): Division ID filter.
            week (int, optional): Week number filter.
            season (str, optional): Season filter.
            **kwargs: Additional request parameters.
        Returns:
            dict: Division ranking data.
        """
        request = {}
        if self.username and self.password:
           request['Credentials'] = self.credentials
        if divisionId:
            request['DivisionId'] = divisionId
        if week:
            request['Week'] = week
        if season:
            request['Season'] = season
        request.update(kwargs)
        if request:
            return self.client.service.GetDivisionRanking(**request)
        else:
            return self.client.service.GetDivisionRanking()

    def getDivisions(self, level: int = None, season: int = None, show_division_name: 'ShowDivisionNameType' = None, **kwargs) -> dict:
        """
        Retrieve division list.
        Args:
            level (int, optional): Level filter.
            season (str, optional): Season filter.
            show_division_name (ShowDivisionNameType, optional): ShowDivisionNameType enum value (required if provided).
            **kwargs: Additional request parameters.
        Returns:
            dict: Divisions data.
        """
        
        request = {}
        if self.username and self.password:
           request['Credentials'] = self.credentials
        if level is not None:
            request['Level'] = level
        if season is not None:
            request['Season'] = season
        if show_division_name is not None:
            request['ShowDivisionName'] = show_division_name.value
        request.update(kwargs)
        if request:
            return self.client.service.GetDivisions(**request)
        else:
            return self.client.service.GetDivisions()

    def getMatches(self, division_id: int = None, club: str = None, team: str = None, division_category: int = None, season: int = None, week_name: str = None, level: int = None, show_division_name: 'ShowDivisionNameType' = None, year_date_from: str = None, year_date_to: str = None, with_details: bool = None, match_id: str = None, match_unique_id: str = None, **kwargs ) -> dict:
        """
        Get list of matches and results. Either provide a Club, or a DivisionId. 
        Args:
            division_id (int, optional): DivisionId filter.
            club (str, optional): Club filter.
            team (str, optional): Team filter.
            division_category (int, optional): DivisionCategory filter.
            season (int, optional): Season filter.
            week_name (str, optional): WeekName filter.
            level (int, optional): Level filter.
            show_division_name (ShowDivisionNameType, optional): ShowDivisionNameType enum value.
            year_date_from (str, optional): YearDateFrom (YYYY-MM-DD).
            year_date_to (str, optional): YearDateTo (YYYY-MM-DD).
            with_details (bool, optional): WithDetails flag.
            match_id (str, optional): MatchId filter.
            match_unique_id (str, optional): MatchUniqueId filter.
            **kwargs: Additional request parameters.
        Returns:
            dict: Matches data.
        """
        if club is None and division_id is None:
            raise ValueError("Either 'club' or 'division_id' must be provided to getMatches.")

        request = {}
        if self.username and self.password:
            request['Credentials'] = self.credentials
        if division_id is not None:
            request['DivisionId'] = division_id
        if club is not None:
            request['Club'] = club
        if team is not None:
            request['Team'] = team
        if division_category is not None:
            request['DivisionCategory'] = division_category
        if season is not None:
            request['Season'] = season
        if week_name is not None:
            request['WeekName'] = week_name
        if level is not None:
            request['Level'] = level
        if show_division_name is not None:
            request['ShowDivisionName'] = show_division_name.value
        if year_date_from is not None:
            request['YearDateFrom'] = year_date_from
        if year_date_to is not None:
            request['YearDateTo'] = year_date_to
        if with_details is not None:
            request['WithDetails'] = with_details
        if match_id is not None:
            request['MatchId'] = match_id
        if match_unique_id is not None:
            request['MatchUniqueId'] = match_unique_id
        request.update(kwargs)
        if request:
            return self.client.service.GetMatches(**request)
        else:
            return self.client.service.GetMatches()

    def getMatchSystems(self, **kwargs):
        """
        Retrieve list of match systems.
        Args:
            **kwargs: Additional request parameters.
        Returns:
            dict: Match systems data.
        """
        request = {}
        request.update(kwargs)
        if request:
            return self.client.service.GetMatchSystems(**request)
        else:
            return self.client.service.GetMatchSystems()

    def getMembers(self, club: str = None, season: int = None, player_category: int = None, unique_index: int = None, name_search: str = None, extended_information: bool = None, ranking_points_information: bool = None, with_results: bool = None, with_opponent_ranking_evaluation: bool = None, **kwargs) -> dict:
        """
        Get list of members according to search criteria.

        Args:
            club (str, optional): Club ID filter.
            season (int, optional): Season filter.
            player_category (int, optional): PlayerCategory filter.
            unique_index (int, optional): UniqueIndex filter.
            name_search (str, optional): NameSearch filter.
            extended_information (bool, optional): ExtendedInformation flag.
            ranking_points_information (bool, optional): RankingPointsInformation flag.
            with_results (bool, optional): WithResults flag.
            with_opponent_ranking_evaluation (bool, optional): WithOpponentRankingEvaluation flag.
            **kwargs: Additional filters.

        Returns:
            dict: Members data.
        """
        request = {}
        if self.username and self.password:
            request['Credentials'] = self.credentials
        if club is not None:
            request['Club'] = club
        if season is not None:
            request['Season'] = season
        if player_category is not None:
            request['PlayerCategory'] = player_category
        if unique_index is not None:
            request['UniqueIndex'] = unique_index
        if name_search is not None:
            request['NameSearch'] = name_search
        if extended_information is not None:
            request['ExtendedInformation'] = extended_information
        if ranking_points_information is not None:
            request['RankingPointsInformation'] = ranking_points_information
        if with_results is not None:
            request['WithResults'] = with_results
        if with_opponent_ranking_evaluation is not None:
            request['WithOpponentRankingEvaluation'] = with_opponent_ranking_evaluation
        request.update(kwargs)
        if request:
            return self.client.service.GetMembers(**request)
        else:
            return self.client.service.GetMembers()

    def getTournaments(self, season: int = None, tournament_unique_index: int = None, with_results: bool = None, with_registrations: bool = None,  **kwargs    ) -> dict:
        """
        Get tournaments according to search criteria.

        Args:
            season (int, optional): Season filter.
            tournament_unique_index (int, optional): TournamentUniqueIndex filter.
            with_results (bool, optional): WithResults flag.
            with_registrations (bool, optional): WithRegistrations flag.
            **kwargs: Additional filters.

        Returns:
            dict: Tournament data.
        """
        request = {}
        if self.username and self.password:
            request['Credentials'] = self.credentials
        if season is not None:
            request['Season'] = season
        if tournament_unique_index is not None:
            request['TournamentUniqueIndex'] = tournament_unique_index
        if with_results is not None:
            request['WithResults'] = with_results
        if with_registrations is not None:
            request['WithRegistrations'] = with_registrations
        request.update(kwargs)
        if request:
            return self.client.service.GetTournaments(**request)
        else:
            return self.client.service.GetTournaments()

    def getPlayerCategories(self, season: int = None, unique_index: int = None, short_name_search: str = None, ranking_category: int = None, **kwargs) -> dict:
        """
        Get player categories according to search criteria.

        Args:
            season (int, optional): Season filter.
            unique_index (int, optional): UniqueIndex filter.
            short_name_search (str, optional): ShortNameSearch filter.
            ranking_category (int, optional): RankingCategory filter.
            **kwargs: Additional filters.

        Returns:
            dict: Player categories data.
        """
        request = {}
        if self.username and self.password:
            request['Credentials'] = self.credentials
        if season is not None:
            request['Season'] = season
        if unique_index is not None:
            request['UniqueIndex'] = unique_index
        if short_name_search is not None:
            request['ShortNameSearch'] = short_name_search
        if ranking_category is not None:
            request['RankingCategory'] = ranking_category
        request.update(kwargs)
        if request:
            return self.client.service.GetPlayerCategories(**request)
        else:
            return self.client.service.GetPlayerCategories()

    def upload(self, *args, **kwargs):
        """
        Placeholder for Upload function (not implemented in read-only version).
        """
        raise NotImplementedError("Upload is not supported in read-only version.")

    def tournamentRegister(self, *args, **kwargs):
        """
        Placeholder for TournamentRegister function (not implemented in read-only version).
        """
        raise NotImplementedError("TournamentRegister is not supported in read-only version.")
