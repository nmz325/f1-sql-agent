{#
  Converts a lap time string formatted as m:ss.fff (for example '1:27.452')
  into integer milliseconds. NULL or empty strings return NULL; any other
  format raises an error, so bad data fails the build instead of becoming
  a silent NULL.
#}
{% macro lap_time_to_ms(column) -%}
    case
        when nullif(trim({{ column }}), '') is null then null
        when regexp_full_match(trim({{ column }}), '\d+:\d{2}\.\d+')
            then cast(split_part(trim({{ column }}), ':', 1) as integer) * 60000
                + cast(round(cast(split_part(trim({{ column }}), ':', 2) as double) * 1000) as integer)
        else cast(error('Unparseable lap time: ' || {{ column }}) as integer)
    end
{%- endmacro %}
