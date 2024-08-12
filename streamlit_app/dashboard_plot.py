# dashboard_plot.py

import plotly.graph_objects as go
from params import *
import pandas as pd


def plot_dashboard(tss, atl, ctl, tsb):
    tss = tss.interpolate()
    merged_df = ctl.join(atl).join(tsb)
    merged_df.columns = ['CTL', 'ATL', 'TSB']

    # Convert the date strings to datetime objects
    start_date = pd.to_datetime('2023-03-03')
    end_date = tss.index.max()
    given_date = GIVEN_DATE  # Example date, update as needed

    # Define colors
    ctl_line_color = 'rgba(0, 0, 255, 0.8)'  # Dark blue color for the line
    fill_color_before = 'rgba(0, 0, 255, 0.2)'  # More transparent blue for the fill before the given_date
    fill_color_after = 'rgba(0, 0, 255, 0.1)'  # Even lighter blue for the fill after the given_date

    # Split data into before and after the given_date
    ctl_before = ctl[ctl.index <= given_date]
    ctl_after = ctl[ctl.index > given_date]
    atl_before = atl[atl.index <= given_date]
    atl_after = atl[atl.index > given_date]
    tsb_before = tsb[tsb.index <= given_date]
    tsb_after = tsb[tsb.index > given_date]

    # Create traces for each metric with date-based line styling
    fig = go.Figure()

    # Add CTL Trace (before given_date)
    fig.add_trace(go.Scatter(
        x=ctl_before.index.tolist(),
        y=ctl_before['CTL'],
        mode='lines',
        name='CTL',
        fill='tozeroy',  # Fill to the y-axis
        fillcolor=fill_color_before,  # More transparent blue
        line=dict(width=2, color=ctl_line_color),  # Thinner line, darker blue
        hovertemplate='<b>Date</b>: %{x}<br><b>CTL</b>: %{y:.2f}<br>' +
                      '<b>ATL</b>: %{customdata[1]:.2f}<br>' +
                      '<b>TSB</b>: %{customdata[2]:.2f}<extra></extra>',
        customdata=merged_df.loc[ctl_before.index, ['CTL', 'ATL', 'TSB']].values,
        showlegend=True  # Ensure legend shows for CTL
    ))

    # Add CTL Trace (after given_date)
    fig.add_trace(go.Scatter(
        x=ctl_after.index.tolist(),
        y=ctl_after['CTL'],
        mode='lines',
        name='CTL',
        fill='tozeroy',  # Fill to the y-axis
        fillcolor=fill_color_after,  # Even lighter blue
        line=dict(width=2, color=ctl_line_color, dash='dash'),  # Thinner dashed line, darker blue
        hovertemplate='<b>Date</b>: %{x}<br><b>CTL</b>: %{y:.2f}<br>' +
                      '<b>ATL</b>: %{customdata[1]:.2f}<br>' +
                      '<b>TSB</b>: %{customdata[2]:.2f}<extra></extra>',
        customdata=merged_df.loc[ctl_after.index, ['CTL', 'ATL', 'TSB']].values,
        showlegend=False  # Avoid duplicate legend entries
    ))

    # Add ATL Trace (before given_date)
    fig.add_trace(go.Scatter(
        x=atl_before.index.tolist(),
        y=atl_before['ATL'],
        mode='lines',
        name='ATL',
        line=dict(width=2, color='magenta'),
        hovertemplate='<b>Date</b>: %{x}<br><b>ATL</b>: %{y:.2f}<br>' +
                      '<b>CTL</b>: %{customdata[0]:.2f}<br>' +
                      '<b>TSB</b>: %{customdata[2]:.2f}<extra></extra>',
        customdata=merged_df.loc[atl_before.index, ['CTL', 'ATL', 'TSB']].values,
        showlegend=True  # Ensure legend shows for ATL
    ))

    # Add ATL Trace (after given_date)
    fig.add_trace(go.Scatter(
        x=atl_after.index.tolist(),
        y=atl_after['ATL'],
        mode='lines',
        name='ATL',
        line=dict(width=2, color='magenta', dash='dash'),
        hovertemplate='<b>Date</b>: %{x}<br><b>ATL</b>: %{y:.2f}<br>' +
                      '<b>CTL</b>: %{customdata[0]:.2f}<br>' +
                      '<b>TSB</b>: %{customdata[2]::.2f}<extra></extra>',
        customdata=merged_df.loc[atl_after.index, ['CTL', 'ATL', 'TSB']].values,
        showlegend=False  # Avoid duplicate legend entries
    ))

    # Add TSB Trace (before given_date)
    fig.add_trace(go.Scatter(
        x=tsb_before.index.tolist(),
        y=tsb_before['TSB'],
        mode='lines',
        name='TSB',
        line=dict(width=2, color='orange'),
        hovertemplate='<b>Date</b>: %{x}<br><b>TSB</b>: %{y:.2f}<br>' +
                      '<b>CTL</b>: %{customdata[0]:.2f}<br>' +
                      '<b>ATL</b>: %{customdata[1]:.2f}<extra></extra>',
        customdata=merged_df.loc[tsb_before.index, ['CTL', 'ATL', 'TSB']].values,
        showlegend=True  # Ensure legend shows for TSB
    ))

    # Add TSB Trace (after given_date)
    fig.add_trace(go.Scatter(
        x=tsb_after.index.tolist(),
        y=tsb_after['TSB'],
        mode='lines',
        name='TSB',
        line=dict(width=2, color='orange', dash='dash'),
        hovertemplate='<b>Date</b>: %{x}<br><b>TSB</b>: %{y:.2f}<br>' +
                      '<b>CTL</b>: %{customdata[0]:.2f}<br>' +
                      '<b>ATL</b>: %{customdata[1]:.2f}<extra></extra>',
        customdata=merged_df.loc[tsb_after.index, ['CTL', 'ATL', 'TSB']].values,
        showlegend=False  # Avoid duplicate legend entries
    ))

    # Define layout
    layout = go.Layout(
        title='Performance Management - Workout Type: All Workout Types',
        titlefont=dict(size=24, family='Arial', color='black', weight='bold'),
        xaxis=dict(
            title='Date',
            range=[start_date, end_date],
            tickangle=45,
            showgrid=True,
            zeroline=True,
            tickformat='%b %d'  # Format the date to show only month and day
        ),
        yaxis=dict(
            title='Form (TSB)',
            side='right',
            showgrid=True,
            zeroline=True,
            titlefont=dict(color='orange'),
            tickfont=dict(color='orange')
        ),
        legend=dict(
            title='Metrics',
            orientation='h',
            x=0,
            y=1.1,
            traceorder='normal',
            itemsizing='constant',
            itemclick='toggleothers',  # Click behavior for the legend
            itemdoubleclick='toggle'   # Double-click behavior for the legend
        ),
        height=600,
        width=1000,
        template='plotly_white'
    )

    # Update layout
    fig.update_layout(layout)

    return fig
