# Copyright (c) 2022 Jaime van Kessel
# The SmartFit Orientation plugin is released under the terms of the LGPLv3 or higher.
# AI-assisted adjustments applied.

from UM.i18n import i18nCatalog
i18n_catalog = i18nCatalog("SmartFitOrientation")

from . import SmartFitOrientation

def getMetaData():
    return {}


def register(app):
    return {"extension": SmartFitOrientation.SmartFitOrientationPlugin()}
