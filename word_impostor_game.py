import streamlit as st
import random
from datetime import datetime

# Initialize session state
if 'game_state' not in st.session_state:
    st.session_state.game_state = 'setup'
if 'players' not in st.session_state:
    st.session_state.players = []
if 'words' not in st.session_state:
    st.session_state.words = [
        ("Kat", "Levend"),
        ("Hond", "Levend"),
        ("Koffie", "Warm"),
        ("Thee", "Warm"),
        ("Pizza", "Eetbaar"),
        ("Brood", "Eetbaar"),
        ("Zomer", "Periode"),
        ("Winter", "Periode"),
        ("Boek", "Papier"),
        ("Pen", "Klein"),
        ("Oceaan", "Groot"),
        ("Berg", "Groot"),
        ("Appel", "Rond"),
        ("Banaan", "Gebogen"),
        ("Gitaar", "Geluid"),
        ("Piano", "Geluid"),
        ("Voetbal", "Beweging"),
        ("Tennis", "Beweging"),
        ("Zonsopgang", "Licht"),
        ("Middernacht", "Donker"),
        ("Auto", "Snel"),
        ("Fiets", "Langzaam"),
        ("Telefoon", "Verbinding"),
        ("Computer", "Scherm"),
        ("Boom", "Groen"),
        ("Bloem", "Kleurrijk"),
        ("Maan", "Nacht"),
        ("Ster", "Nacht"),
        ("Vuur", "Heet"),
        ("Water", "Nat"),
        ("Stoel", "Zitplek"),
        ("Tafel", "Oppervlak"),
        ("Rood", "Zichtbaar"),
        ("Blauw", "Zichtbaar"),
        ("School", "Druk"),
        ("Ziekenhuis", "Stil"),
        ("Dokter", "Helpen"),
        ("Leraar", "Uitleggen"),
        ("Huis", "Binnen"),
        ("Appartement", "Hoog")
    ]
if 'current_word' not in st.session_state:
    st.session_state.current_word = None
if 'current_hint' not in st.session_state:
    st.session_state.current_hint = None
if 'impostor_indices' not in st.session_state:
    st.session_state.impostor_indices = []
if 'revealed_players' not in st.session_state:
    st.session_state.revealed_players = set()

st.title("🎭 Word Impostor Game")
st.markdown("---")

# Setup Phase
if st.session_state.game_state == 'setup':
    st.header("Game Setup")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Add Players")
        player_name = st.text_input("Player Name", key="player_input")
        if st.button("Add Player") and player_name:
            if player_name not in st.session_state.players:
                st.session_state.players.append(player_name)
                st.rerun()
            else:
                st.warning("Player already added!")
    
    with col2:
        st.subheader("Current Players")
        if st.session_state.players:
            for i, player in enumerate(st.session_state.players):
                cols = st.columns([4, 1])
                cols[0].write(f"{i+1}. {player}")
                if cols[1].button("❌", key=f"remove_{i}"):
                    st.session_state.players.pop(i)
                    st.rerun()
        else:
            st.info("No players yet. Add at least 3 players to start!")
    
    st.markdown("---")
    
    if len(st.session_state.players) >= 3:
        max_impostors = max(1, len(st.session_state.players) // 3)
        num_impostors = st.slider("Number of Impostors", 1, max_impostors, 1) if max_impostors > 1 else 1
        
        if st.button("🎮 Start Game", type="primary"):
            # Select random word with hint
            word_pair = random.choice(st.session_state.words)
            st.session_state.current_word = word_pair[0]
            st.session_state.current_hint = word_pair[1]
            # Select random impostors
            st.session_state.impostor_indices = random.sample(
                range(len(st.session_state.players)), 
                num_impostors
            )
            st.session_state.game_state = 'playing'
            st.session_state.revealed_players = set()
            st.rerun()
    else:
        st.warning("⚠️ Need at least 3 players to start the game!")

# Playing Phase
elif st.session_state.game_state == 'playing':
    st.header("🎮 Game in Progress")
    
    st.info("Each player should view their word privately. Click your name to reveal your word!")
    
    word = st.session_state.current_word
    hint = st.session_state.current_hint
    
    for i, player in enumerate(st.session_state.players):
        is_impostor = i in st.session_state.impostor_indices
        
        with st.expander(f"👤 {player}'s Word", expanded=False):
            if i not in st.session_state.revealed_players:
                if st.button(f"🔍 Reveal Word for {player}", key=f"reveal_{i}"):
                    st.session_state.revealed_players.add(i)
                    st.rerun()
            else:
                if is_impostor:
                    st.error(f"### 🎭 You are the IMPOSTOR!")
                    st.markdown(f"**Hint:** The word is a type of **{hint}**")
                    st.markdown("*Listen carefully to others and try to blend in!*")
                else:
                    st.success(f"### Your word is: **{word}**")
                    st.markdown("*Remember: Don't say your word directly! Describe it carefully.*")
                if st.button(f"Hide", key=f"hide_{i}"):
                    st.session_state.revealed_players.remove(i)
                    st.rerun()
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 New Game"):
            st.session_state.game_state = 'setup'
            st.session_state.revealed_players = set()
            st.rerun()
    
    with col2:
        if st.button("👁️ Reveal All (End Game)"):
            st.session_state.game_state = 'reveal'
            st.rerun()

# Reveal Phase
elif st.session_state.game_state == 'reveal':
    st.header("🎭 Game Results")
    
    word = st.session_state.current_word
    
    st.subheader(f"The Word Was: **{word}**")
    
    st.markdown("---")
    st.subheader("Players:")
    
    for i, player in enumerate(st.session_state.players):
        is_impostor = i in st.session_state.impostor_indices
        if is_impostor:
            st.error(f"🎭 **{player}** - IMPOSTOR (didn't know the word)")
        else:
            st.success(f"✅ **{player}** - Regular (had word: {word})")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Play Again"):
            st.session_state.game_state = 'setup'
            st.session_state.revealed_players = set()
            st.rerun()
    with col2:
        if st.button("🎮 New Game (Same Players)"):
            # Select new word with hint
            word_pair = random.choice(st.session_state.words)
            st.session_state.current_word = word_pair[0]
            st.session_state.current_hint = word_pair[1]
            # Select new impostors
            num_impostors = len(st.session_state.impostor_indices)
            st.session_state.impostor_indices = random.sample(
                range(len(st.session_state.players)), 
                num_impostors
            )
            st.session_state.game_state = 'playing'
            st.session_state.revealed_players = set()
            st.rerun()

# Sidebar with rules
with st.sidebar:
    st.header("📖 How to Play")
    st.markdown("""
    1. **Setup**: Add 3+ players
    2. **Start**: Most players get a word, impostor(s) don't
    3. **Impostors**: Don't know the word, only get a hint
    4. **Gameplay**: 
       - Take turns describing your word
       - Don't say the word directly!
       - Try to figure out who's the impostor
    5. **Vote**: Discuss and vote out the impostor
    6. **Win**: 
       - Crew wins if they find the impostor
       - Impostor wins if they blend in
    
    **Tips:**
    - Regular players: Be specific but not too obvious
    - Watch for vague or off-topic descriptions
    - Impostors: listen carefully and adapt!
    """)
