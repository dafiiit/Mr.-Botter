import numpy as np
from datetime import datetime


def time_to_sin_cos_vector(dt):
    """
    Nimmt ein datetime-Objekt und gibt die Sinus- und Cosinuswerte für Sekunden, Minuten, Stunden, Tage, Monate und Jahre als Vektor zurück. Sin und Cos zusammen repräsentieren den Punkt auf dem Kreis.
    """

    # Sekunden
    sec = dt.second
    sec_sin = np.sin(2 * np.pi * (sec / 60))
    sec_cos = np.cos(2 * np.pi * (sec / 60))

    # Minuten
    minute = dt.minute
    min_sin = np.sin(2 * np.pi * (minute / 60))
    min_cos = np.cos(2 * np.pi * (minute / 60))

    # Stunden
    hour = dt.hour
    hour_sin = np.sin(2 * np.pi * (hour / 24))
    hour_cos = np.cos(2 * np.pi * (hour / 24))

    # Tage
    day = dt.day
    days_in_month = (
        datetime(dt.year, dt.month % 12 + 1, 1) - datetime(dt.year, dt.month, 1)
    ).days
    day_sin = np.sin(2 * np.pi * ((day - 1) / days_in_month))
    day_cos = np.cos(2 * np.pi * ((day - 1) / days_in_month))

    # Monate
    month = dt.month
    month_sin = np.sin(2 * np.pi * ((month - 1) / 12))
    month_cos = np.cos(2 * np.pi * ((month - 1) / 12))

    # Jahre
    year_day = dt.timetuple().tm_yday
    days_in_year = (
        366
        if (dt.year % 4 == 0 and (dt.year % 100 != 0 or dt.year % 400 == 0))
        else 365
    )
    year_sin = np.sin(2 * np.pi * ((year_day - 1) / days_in_year))
    year_cos = np.cos(2 * np.pi * ((year_day - 1) / days_in_year))

    # Vektor der Sinus- und Cosinuswerte
    time_vector = np.array(
        [
            sec_sin,
            sec_cos,
            min_sin,
            min_cos,
            hour_sin,
            hour_cos,
            day_sin,
            day_cos,
            month_sin,
            month_cos,
            year_sin,
            year_cos,
        ]
    )

    return time_vector

# Beispiel: Jetzt
now = datetime.now()
time_encoding_vector = time_to_sin_cos_vector(now)
print(time_encoding_vector)
