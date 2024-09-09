import streamlit as st

from menu import menu

st.set_page_config(page_title="Impressum", page_icon=None, layout="wide", initial_sidebar_state="auto")
st.session_state.switched_page = True

st.title("Impressum")

st.markdown(
    '''
    ### Angaben gem. § 5 TMG:  

    Max Muster  
    Musterweg  
    12345 Musterstadt  

    ### Vertreten durch:  
    Max Muster  

    ### Kontakt:  
    Telefon: 01234-789456  
    Fax: 1234-56789  
    E-Mail: max@muster.de  

    ### Registereintrag:  
    Eintragung im Registergericht: Musterstadt  
    Registernummer: 12345  

    ### Umsatzsteuer-ID:  
    Umsatzsteuer-Identifikationsnummer gemäß §27a Umsatzsteuergesetz: DE124356789  

    ### Wirtschafts-ID:  
    DE123456789-00001

    ### Aufsichtsbehörde:  
    Musteraufsicht Musterstadt  

    '''
)


menu()