// Copyright (c) 2022 Jaime van Kessel
// Part of SmartFit Orientation. Licensed under LGPL-3.0. See LICENSE.

import QtQuick 2.2
import QtQuick.Controls 2.0

import UM 1.2 as UM

UM.Dialog
{
    minimumWidth: 450
    minimumHeight: 140
    function boolCheck(value) //Hack to ensure a good match between python and qml.
    {
        if(value == "True")
        {
            return true
        }else if(value == "False" || value == undefined)
        {
            return false
        }
        else
        {
            return value
        }
    }

    title: "SmartFit Orientation settings"

    Column
    {
        spacing: 12
        width: parent.width - 20

        CheckBox
        {
            width: parent.width - parent.spacing
            checked: boolCheck(UM.Preferences.getValue("SmartFitOrientation/do_auto_orientation"))
            onClicked: UM.Preferences.setValue("SmartFitOrientation/do_auto_orientation", checked)

            text: "Automatically calculate the orientation for all loaded models"
        }

        Row
        {
            spacing: 4
            width: parent.width - parent.spacing

            CheckBox
            {
                width: parent.width - 28
                checked: boolCheck(UM.Preferences.getValue("SmartFitOrientation/fast_fit_check"))
                onClicked: UM.Preferences.setValue("SmartFitOrientation/fast_fit_check", checked)

                text: "Use fast fit check (faster; uncheck for more precise build-volume check)"
            }

            Item
            {
                width: 24
                height: 24
                ToolTip
                {
                    visible: fastFitHelpMouse.containsMouse
                    delay: 300
                    text: "Checked: faster build-volume check (5° steps, stops when one angle fits). Unchecked: precise check (0.5° over all angles, slower but more thorough). Final angle is always found precisely."
                    font.pixelSize: 11
                    palette.toolTipText: "#1a1a1a"
                }
                Text
                {
                    text: "?"
                    font.pixelSize: 14
                    color: UM.Theme.getColor("text")
                    anchors.centerIn: parent
                }
                MouseArea
                {
                    id: fastFitHelpMouse
                    anchors.fill: parent
                    hoverEnabled: true
                }
            }
        }
    }
}